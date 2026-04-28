"""LSTM 기반 시계열 예측 모델"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset


class LSTMNetwork(nn.Module):
    """LSTM 신경망"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # 3 classes: down, neutral, up
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class LSTMPredictor:
    """LSTM 시계열 패턴 → 방향 예측"""

    # 메모리 보호: 최대 시퀀스 수 제한 (OOM 방지)
    MAX_SEQUENCES = 80000
    # 5분봉을 1시간봉으로 다운샘플링하는 배수
    DOWNSAMPLE_RATIO = 12

    def __init__(self, seq_length: int = 60, model_dir: str = "models_saved"):
        self.seq_length = seq_length
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: LSTMNetwork | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns: list[str] = []
        self.accuracy: float = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """시퀀스 데이터 생성 (메모리 최적화: 최대 시퀀스 수 제한)

        1. 데이터가 많으면 1시간봉으로 다운샘플링 (5분봉 × 12)
        2. 그래도 많으면 랜덤 서브샘플링 (최대 MAX_SEQUENCES)
        """
        total_n = len(X) - self.seq_length

        # 1단계: 데이터가 너무 많으면 1시간봉으로 다운샘플링
        if total_n > self.MAX_SEQUENCES * self.DOWNSAMPLE_RATIO:
            X = X[::self.DOWNSAMPLE_RATIO]
            y = y[::self.DOWNSAMPLE_RATIO]
            total_n = len(X) - self.seq_length
            logger.info(
                f"LSTM: 1시간봉 다운샘플링 적용 (5분×{self.DOWNSAMPLE_RATIO}) → "
                f"{len(X):,}개 캔들"
            )

        # 2단계: 그래도 많으면 랜덤 서브샘플링
        if total_n > self.MAX_SEQUENCES:
            rng = np.random.default_rng(seed=42)
            indices = np.sort(rng.choice(total_n, self.MAX_SEQUENCES, replace=False))
            logger.info(
                f"LSTM: 시퀀스 서브샘플링 {total_n:,} → {self.MAX_SEQUENCES:,}개"
            )
        else:
            indices = np.arange(total_n)

        # 시퀀스 생성 (float32로 메모리 절약)
        Xs = np.stack([X[i:i + self.seq_length] for i in indices]).astype(np.float32)
        ys = y[indices + self.seq_length].astype(np.int64)

        gc.collect()
        return Xs, ys

    def train(self, df: pd.DataFrame, feature_cols: list[str], label_col: str = "label",
              epochs: int = 50, batch_size: int = 64, lr: float = 0.001):
        """모델 학습 (기존 모델이 있으면 fine-tuning, 없으면 최초 학습)"""
        gc.collect()  # 학습 시작 전 메모리 정리

        # === 피처 수 불일치 감지 (2026-04-20 추가) ===
        # 기존 버그: self.feature_columns = feature_cols를 먼저 한 뒤 is_finetune을
        # len(feature_cols) == len(self.feature_columns)로 비교 → 항상 True가 되어
        # 실제 LSTMNetwork.lstm.input_size (아키텍처) 와 다를 때도 fine-tune 시도 → RuntimeError.
        # 수정: 실제 모델의 LSTM input_size와 비교.
        if self.model is not None:
            try:
                prev_input_size = self.model.lstm.input_size
                if prev_input_size != len(feature_cols):
                    logger.warning(
                        f"LSTM 피처 수 불일치 감지: 기존 input_size={prev_input_size} vs "
                        f"신규={len(feature_cols)} → fine-tune 불가, 최초학습으로 전환"
                    )
                    self.model = None
                    self.accuracy = 0.0
            except Exception as e:
                logger.debug(f"LSTM 피처 수 비교 실패 (무시): {e}")

        self.feature_columns = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        # inf/nan 방어 (2026-04-20 추가) — bb_pct, ema_cross 등이 0 나눗셈으로
        # inf를 만들면 std 계산이 inf가 되어 LSTM loss=nan 유발.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 정규화
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std
        # 정규화 후에도 inf/nan 한 번 더 (std=0인 상수 컬럼 방어)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_seq, y_seq = self._create_sequences(X, y)
        # 원본 X, y는 더 이상 필요 없음 — 메모리 해제
        del X, y
        gc.collect()

        split = int(len(X_seq) * 0.8)

        train_ds = TensorDataset(
            torch.FloatTensor(X_seq[:split]),
            torch.LongTensor(y_seq[:split]),
        )
        test_X = torch.FloatTensor(X_seq[split:]).to(self.device)
        test_y = torch.LongTensor(y_seq[split:]).to(self.device)

        # numpy 배열 삭제 (텐서로 복사 완료)
        del X_seq, y_seq
        gc.collect()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        prev_accuracy = self.accuracy
        is_finetune = self.model is not None and len(feature_cols) == len(self.feature_columns)

        if is_finetune:
            # === Fine-tuning: 기존 가중치에서 이어서 학습 ===
            logger.info(f"LSTM Fine-tuning 시작 (기존 정확도: {prev_accuracy:.4f})")
            finetune_lr = lr * 0.3  # 학습률 30%로 낮춰서 기존 지식 보존
            finetune_epochs = max(epochs // 2, 15)  # 에폭 절반
            optimizer = torch.optim.Adam(self.model.parameters(), lr=finetune_lr)

            # 기존 모델의 best state 백업 (롤백용)
            best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            # === 최초 학습 ===
            logger.info("LSTM 최초 학습 시작")
            finetune_epochs = epochs
            self.model = LSTMNetwork(input_size=len(feature_cols)).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            best_state = None

        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(finetune_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
                optimizer.step()
                total_loss += loss.item()

            # 검증
            self.model.eval()
            with torch.no_grad():
                pred = self.model(test_X).argmax(dim=1)
                acc = (pred == test_y).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                train_type = "Fine-tune" if is_finetune else "Train"
                logger.info(f"LSTM {train_type} {epoch+1}/{finetune_epochs} - Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")

        # 성능 하락 시 롤백
        if is_finetune and prev_accuracy > 0 and best_acc < prev_accuracy - 0.05:
            logger.warning(
                f"LSTM 정확도 하락 감지: {prev_accuracy:.4f} → {best_acc:.4f} — 기존 모델 유지"
            )
            if best_state:
                self.model.load_state_dict(best_state)
            self.accuracy = prev_accuracy
            return self.accuracy

        # best state 복원
        if best_state:
            self.model.load_state_dict(best_state)

        self.accuracy = best_acc
        train_type = "Fine-tune" if is_finetune else "최초학습"
        improvement = f" ({best_acc - prev_accuracy:+.4f})" if prev_accuracy > 0 else ""
        logger.info(f"LSTM {train_type} 완료 - 최고 정확도: {self.accuracy:.4f}{improvement}")

        # 학습 완료 후 메모리 정리
        del train_ds, test_X, test_y, train_loader
        gc.collect()

        return self.accuracy

    def train_walkforward(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
        n_splits: int = 3,
        purge_gap: int = 12,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
    ) -> float:
        """Walk-forward CV for LSTM (tier=small+ 활성화)

        LSTM은 학습 비용이 커서 n_splits=3 기본. 각 폴드 epochs 축소(30).
        최종 모델은 마지막 폴드(가장 많은 데이터) 기준.
        """
        gc.collect()
        self.feature_columns = feature_cols
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].values.astype(np.int64)

        # 정규화
        self.mean = X_raw.mean(axis=0)
        self.std = X_raw.std(axis=0) + 1e-8
        X = (X_raw - self.mean) / self.std

        # 시퀀스 생성 전 인덱스 기준 walk-forward split
        n_samples = len(X) - self.seq_length
        if n_samples < n_splits * 500:
            logger.warning(
                f"[WalkForward-LSTM] 샘플 부족 ({n_samples} < {n_splits*500}) → 기존 train()로 fallback"
            )
            return self.train(df, feature_cols, label_col, epochs=epochs, batch_size=batch_size, lr=lr)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accs = []
        prev_accuracy = self.accuracy

        logger.info(f"[WalkForward-LSTM] {n_splits}-fold CV 시작 (purge_gap={purge_gap}, epochs={epochs})")

        indices = np.arange(n_samples)
        last_state = None

        for fold_idx, (train_idx_raw, test_idx_raw) in enumerate(tscv.split(indices)):
            # Purge
            if len(train_idx_raw) > purge_gap:
                train_idx_raw = train_idx_raw[:-purge_gap]

            # 시퀀스 생성 (폴드별)
            X_tr_seq = np.stack([X[i:i + self.seq_length] for i in train_idx_raw]).astype(np.float32)
            y_tr_seq = y_raw[train_idx_raw + self.seq_length].astype(np.int64)

            X_te_seq = np.stack([X[i:i + self.seq_length] for i in test_idx_raw]).astype(np.float32)
            y_te_seq = y_raw[test_idx_raw + self.seq_length].astype(np.int64)

            # 메모리 상한
            if len(X_tr_seq) > self.MAX_SEQUENCES:
                rng = np.random.default_rng(seed=42 + fold_idx)
                sel = np.sort(rng.choice(len(X_tr_seq), self.MAX_SEQUENCES, replace=False))
                X_tr_seq = X_tr_seq[sel]
                y_tr_seq = y_tr_seq[sel]

            train_ds = TensorDataset(torch.FloatTensor(X_tr_seq), torch.LongTensor(y_tr_seq))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_X_t = torch.FloatTensor(X_te_seq).to(self.device)
            test_y_t = torch.LongTensor(y_te_seq).to(self.device)

            del X_tr_seq, y_tr_seq, X_te_seq, y_te_seq
            gc.collect()

            # 새 모델 (폴드마다 독립 학습 — OOS 정확도의 진짜 추정)
            fold_model = LSTMNetwork(input_size=len(feature_cols)).to(self.device)
            optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_fold_acc = 0.0
            best_fold_state = None
            for epoch in range(epochs):
                fold_model.train()
                for bX, by in train_loader:
                    bX, by = bX.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = fold_model(bX)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer.step()

                fold_model.eval()
                with torch.no_grad():
                    pred = fold_model(test_X_t).argmax(dim=1)
                    acc = (pred == test_y_t).float().mean().item()
                    if acc > best_fold_acc:
                        best_fold_acc = acc
                        best_fold_state = {k: v.clone() for k, v in fold_model.state_dict().items()}

            fold_accs.append(best_fold_acc)
            last_state = best_fold_state  # 마지막 폴드 state 보존 (최종 모델용)
            logger.info(
                f"[WalkForward-LSTM] Fold {fold_idx+1}/{n_splits}: "
                f"train_seq={len(train_loader.dataset)} test_seq={len(test_y_t)} "
                f"best_acc={best_fold_acc:.4f}"
            )

            del fold_model, train_ds, train_loader, test_X_t, test_y_t
            if best_fold_state is not last_state:
                del best_fold_state
            gc.collect()

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        logger.info(
            f"[WalkForward-LSTM] 완료: 평균 acc={mean_acc:.4f} ± {std_acc:.4f} "
            f"(최저={min(fold_accs):.4f}, 최고={max(fold_accs):.4f})"
        )

        # 롤백 가드
        if prev_accuracy > 0 and mean_acc < prev_accuracy - 0.05:
            logger.warning(
                f"[WalkForward-LSTM] OOS {mean_acc:.4f} < 기존 {prev_accuracy:.4f} - 0.05 → 롤백"
            )
            self.accuracy = prev_accuracy
            return self.accuracy

        # 최종 모델: 마지막 폴드의 state 사용
        self.model = LSTMNetwork(input_size=len(feature_cols)).to(self.device)
        if last_state:
            self.model.load_state_dict(last_state)
        self.accuracy = mean_acc
        return mean_acc

    def predict(self, df: pd.DataFrame) -> dict:
        """예측"""
        if self.model is None or self.mean is None:
            return {"signal": 0.0, "confidence": 0.0, "direction": "neutral"}

        # [Patch I, 2026-04-28] 학습 시 추가된 ext_* 누락 컬럼 보정
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            if not getattr(self, "_missing_logged", False):
                logger.warning(
                    f"[LSTM] 누락 피처 {len(missing)}개를 0.0으로 보정 (예: {missing[:5]})"
                )
                self._missing_logged = True
            df = df.copy()
            for c in missing:
                df[c] = 0.0

        X = df[self.feature_columns].values[-self.seq_length:].astype(np.float32)
        # [Patch I, 2026-04-28] NaN/Inf 보정 — 부족한 indicator로 NaN 출력 방지
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # [Patch I] mean/std 가드 — 저장된 normalizer가 NaN/0 가질 수 있음
        # (예: BTC reference 컬럼이 학습 일부 구간에서 NaN → np.mean 결과 NaN)
        mean_safe = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        std_safe = np.nan_to_num(self.std, nan=1.0, posinf=1.0, neginf=1.0)
        std_safe = np.where(std_safe == 0, 1.0, std_safe)
        X = (X - mean_safe) / std_safe
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            proba = torch.softmax(output, dim=1)[0].cpu().numpy()

        pred = int(proba.argmax())
        direction_map = {0: "short", 1: "neutral", 2: "long"}
        signal = float(proba[2] - proba[0])

        return {
            "signal": signal,
            "confidence": float(max(proba)),
            "direction": direction_map[pred],
            "probabilities": {"short": float(proba[0]), "neutral": float(proba[1]), "long": float(proba[2])},
        }

    def save(self, name: str = "lstm"):
        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "features": self.feature_columns,
            "seq_length": self.seq_length,
            "mean": self.mean,
            "std": self.std,
            "accuracy": self.accuracy,
            "input_size": len(self.feature_columns),
        }, path)
        logger.info(f"LSTM 모델 저장: {path}")

    def load(self, name: str = "lstm") -> bool:
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            return False
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_columns = data["features"]
        self.seq_length = data["seq_length"]
        self.mean = data["mean"]
        self.std = data["std"]
        self.accuracy = data["accuracy"]
        self.model = LSTMNetwork(input_size=data["input_size"]).to(self.device)
        self.model.load_state_dict(data["model_state"])
        logger.info(f"LSTM 모델 로드: 정확도 {self.accuracy:.4f}")
        return True
