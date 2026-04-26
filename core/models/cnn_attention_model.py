"""1D-CNN + Multi-Head Attention 시계열 예측 모델 (LSTM 보완)
============================================================
M2 Apple Silicon 최적화:
  - MPS (Metal Performance Shaders) 백엔드 자동 감지/활용
  - 1D-CNN은 LSTM 대비 M2에서 3-10배 빠른 추론
  - 병렬 연산 구조 → Apple SIMD/GPU 최적화

승률 개선:
  - Multi-Scale 1D-CNN (kernel 3/7/15) — 단/중/장기 패턴 동시 포착
  - Squeeze-Excitation — 채널별 중요도 자동 학습
  - Multi-Head Temporal Attention — 핵심 시점에 집중
  - Residual Connection — 안정 학습
  - Label Smoothing + Mixup — 과적합 방지
  - Cosine Annealing LR — 최적 스케줄링

XGBoost/LSTM과 동일한 공개 인터페이스(predict(df), save, load) 제공 →
ensemble.py의 6th vote source로 즉시 통합 가능.

참고: Temporal Attention Mechanisms (ResearchGate 2026)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_optimal_device() -> torch.device:
    """M2 최적 디바이스 자동 감지: MPS > CUDA > CPU"""
    try:
        if torch.backends.mps.is_available():
            logger.info("[CNN-Attn] Apple MPS (Metal) → GPU 가속 활성화")
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        logger.info("[CNN-Attn] CUDA GPU 감지")
        return torch.device("cuda")
    logger.info("[CNN-Attn] CPU 모드")
    return torch.device("cpu")


class SqueezeExcitation(nn.Module):
    """채널별 중요도 학습 (Squeeze-and-Excitation)"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        # reduction이 channels보다 큰 경우 안전 처리
        hidden = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        b, c, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1)
        return x * scale


class TemporalAttention(nn.Module):
    """Multi-Head Temporal Attention"""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class CNNAttentionNet(nn.Module):
    """1D-CNN(Multi-Scale) + SE + Multi-Head Attention 하이브리드"""
    def __init__(
        self, input_dim: int, seq_len: int = 60, n_classes: int = 3,
        d_model: int = 64, n_heads: int = 4, dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        # Multi-Scale CNN — d_model = c_short + c_mid + c_long
        c_short = d_model // 2
        c_mid = d_model // 4
        c_long = d_model - c_short - c_mid  # 합이 d_model 이 되도록 보정
        self.conv_short = nn.Sequential(
            nn.Conv1d(input_dim, c_short, kernel_size=3, padding=1),
            nn.BatchNorm1d(c_short), nn.GELU(),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv1d(input_dim, c_mid, kernel_size=7, padding=3),
            nn.BatchNorm1d(c_mid), nn.GELU(),
        )
        self.conv_long = nn.Sequential(
            nn.Conv1d(input_dim, c_long, kernel_size=15, padding=7),
            nn.BatchNorm1d(c_long), nn.GELU(),
        )
        self.se = SqueezeExcitation(d_model)
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.attention = TemporalAttention(d_model, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) → (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        h = torch.cat([self.conv_short(x), self.conv_mid(x), self.conv_long(x)], dim=1)
        h = self.se(h)
        h = h + self.conv2(h)            # residual
        h = h.permute(0, 2, 1)            # (batch, seq_len, d_model)
        h = self.attention(h)
        h = h.mean(dim=1)                 # GAP over time
        return self.classifier(h)


class CNNAttentionPredictor:
    """CNN+Attention 예측기 — XGBoostPredictor와 동일 인터페이스(predict(df))"""

    def __init__(
        self,
        model_dir: str = "models_saved",
        seq_len: int = 60,
        d_model: int = 64,
        n_heads: int = 4,
        batch_size: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.2,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha

        self.device = get_optimal_device()
        self.model: Optional[CNNAttentionNet] = None
        self.feature_columns: list[str] = []
        self.accuracy: float = 0.0
        self.f1: float = 0.0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        seq, lab = [], []
        for i in range(self.seq_len, len(X)):
            seq.append(X[i - self.seq_len:i])
            lab.append(y[i])
        return np.array(seq, dtype=np.float32), np.array(lab, dtype=np.int64)

    def _eval_chunked(self, X_arr: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """[Patch H] 큰 평가 텐서를 청크 단위로 GPU에 올려 OOM 방지.

        X_arr: (N, seq_len, input_dim) numpy → preds: (N,) int64 numpy
        """
        n = len(X_arr)
        out = np.zeros(n, dtype=np.int64)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            xb = torch.from_numpy(X_arr[i:j]).to(self.device)
            logits = self.model(xb)
            out[i:j] = logits.argmax(dim=1).cpu().numpy()
            del xb, logits
        return out

    def _mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float):
        if alpha <= 0:
            return x, y, y, 1.0
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
        use_mixup: bool = True,
    ):
        """학습 — 라벨 0/1/2 (하락/횡보/상승)"""
        from sklearn.metrics import accuracy_score, f1_score

        self.feature_columns = list(feature_cols)
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score 정규화 (학습 시 통계 기억)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X = (X - self._mean) / self._std

        X_seq, y_seq = self._create_sequences(X, y)
        if len(X_seq) < 100:
            logger.warning(
                f"[CNN-Attn] 시퀀스 부족 ({len(X_seq)} < 100) → 학습 스킵"
            )
            return self.accuracy

        # [Patch H, 2026-04-27] MPS OOM 방지 — 시퀀스 데이터셋이 너무 크면 최근 N개로 cap
        # 691310 캔들 × seq_len × dim × 4byte → MPS 20GiB 한도 초과 사례 발생
        try:
            is_mps = (str(self.device).startswith("mps"))
        except Exception:
            is_mps = False
        if is_mps:
            MPS_MAX_SEQ = 200_000  # 약 4GiB(seq60×dim54×4byte) 기준 안전 한계
            if len(X_seq) > MPS_MAX_SEQ:
                logger.warning(
                    f"[CNN-Attn] MPS 메모리 안전화 — 시퀀스 {len(X_seq):,} → 최근 {MPS_MAX_SEQ:,}개 사용"
                )
                X_seq = X_seq[-MPS_MAX_SEQ:]
                y_seq = y_seq[-MPS_MAX_SEQ:]

        split = int(len(X_seq) * 0.8)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, drop_last=True,
        )

        # 모델 (이전 모델이 있고 input_dim이 다르면 새로 생성)
        prev_acc = self.accuracy
        if self.model is not None:
            try:
                if self.model.input_dim != X_tr.shape[2]:
                    logger.info(
                        f"[CNN-Attn] input_dim 변경 {self.model.input_dim}→{X_tr.shape[2]} → 재생성"
                    )
                    self.model = None
                    prev_acc = 0.0
            except Exception:
                self.model = None
                prev_acc = 0.0

        if self.model is None:
            self.model = CNNAttentionNet(
                input_dim=X_tr.shape[2], seq_len=self.seq_len,
                d_model=self.d_model, n_heads=self.n_heads,
            ).to(self.device)

        # 클래스 가중치
        counts = np.bincount(y_tr, minlength=3).astype(np.float32)
        cw = 1.0 / (counts + 1)
        cw = cw / cw.sum() * 3
        cw_t = torch.from_numpy(cw).to(self.device)

        criterion = nn.CrossEntropyLoss(
            weight=cw_t, label_smoothing=self.label_smoothing,
        )
        optim = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs, eta_min=1e-6,
        )

        best_val = 0.0
        best_state = None
        no_improve = 0
        patience = 10

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                if use_mixup and np.random.random() > 0.5:
                    bx, ya, yb, lam = self._mixup(bx, by, self.mixup_alpha)
                    logits = self.model(bx)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                total_loss += float(loss.item())
            scheduler.step()

            # 검증 — [Patch H] 청크 평가로 MPS OOM 방지
            self.model.eval()
            with torch.no_grad():
                preds = self._eval_chunked(X_te)
                val_acc = float(accuracy_score(y_te, preds))
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"[CNN-Attn] Ep {epoch+1}/{self.epochs} "
                    f"loss={total_loss/max(1,len(loader)):.4f} "
                    f"val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
                )
            if no_improve >= patience:
                logger.info(f"[CNN-Attn] Early stop @ ep{epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 최종 평가 — [Patch H] 청크 평가로 MPS OOM 방지
        self.model.eval()
        with torch.no_grad():
            preds = self._eval_chunked(X_te)
            new_acc = float(accuracy_score(y_te, preds))
            new_f1 = float(f1_score(y_te, preds, average="weighted"))

        # 5%p 하락 시 롤백
        if prev_acc > 0 and new_acc < prev_acc - 0.05:
            logger.warning(
                f"[CNN-Attn] 정확도 하락 {prev_acc:.4f}→{new_acc:.4f} → 기존 모델 유지"
            )
            if self.load():
                return self.accuracy

        self.accuracy = new_acc
        self.f1 = new_f1
        improvement = f" ({new_acc - prev_acc:+.4f})" if prev_acc > 0 else ""
        logger.info(
            f"[CNN-Attn] 학습 완료 — Acc={new_acc:.4f}{improvement} F1={new_f1:.4f}"
        )
        return self.accuracy

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> dict:
        """XGBoostPredictor와 동일한 dict 반환"""
        if self.model is None or self._mean is None:
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "probabilities": {"short": 0.0, "neutral": 0.0, "long": 0.0},
            }
        try:
            X = df[self.feature_columns].values.astype(np.float32)
        except KeyError:
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "probabilities": {"short": 0.0, "neutral": 0.0, "long": 0.0},
            }
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (X - self._mean) / self._std

        # 시퀀스 형태로 변환 (마지막 seq_len개)
        if X.ndim == 2 and X.shape[0] >= self.seq_len:
            X_seq = X[-self.seq_len:].reshape(1, self.seq_len, -1)
        elif X.ndim == 2 and X.shape[0] >= 1:
            # seq_len 부족 시 마지막 행으로 패딩
            pad = np.tile(X[-1:], (self.seq_len - X.shape[0], 1)) if X.shape[0] < self.seq_len else np.zeros((0, X.shape[1]), dtype=np.float32)
            X_seq = np.vstack([pad, X])[-self.seq_len:].reshape(1, self.seq_len, -1)
        else:
            return {
                "signal": 0.0, "confidence": 0.0, "direction": "neutral",
                "probabilities": {"short": 0.0, "neutral": 0.0, "long": 0.0},
            }

        self.model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_seq.astype(np.float32)).to(self.device)
            logits = self.model(xt)
            proba = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(proba))
        direction_map = {0: "short", 1: "neutral", 2: "long"}
        signal = float(proba[2] - proba[0])
        return {
            "signal": signal,
            "confidence": float(proba[pred]),
            "direction": direction_map[pred],
            "probabilities": {
                "short": float(proba[0]),
                "neutral": float(proba[1]),
                "long": float(proba[2]),
            },
        }

    # ------------------------------------------------------------------
    def save(self, name: str = "cnn_attention"):
        if self.model is None:
            return
        path = self.model_dir / f"{name}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "input_dim": self.model.input_dim,
                "seq_len": self.model.seq_len,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
            },
            "feature_columns": self.feature_columns,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "mean": self._mean,
            "std": self._std,
        }, path)
        logger.info(f"[CNN-Attn] 저장: {path}")

    def load(self, name: str = "cnn_attention") -> bool:
        path = self.model_dir / f"{name}.pt"
        if not path.exists():
            return False
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            cfg = data["config"]
            self.model = CNNAttentionNet(
                input_dim=cfg["input_dim"],
                seq_len=cfg["seq_len"],
                d_model=cfg.get("d_model", self.d_model),
                n_heads=cfg.get("n_heads", self.n_heads),
            ).to(self.device)
            self.model.load_state_dict(data["model_state"])
            self.model.eval()
            self.feature_columns = list(data.get("feature_columns") or [])
            self.accuracy = float(data.get("accuracy", 0.0))
            self.f1 = float(data.get("f1", 0.0))
            self._mean = data.get("mean")
            self._std = data.get("std")
            logger.info(f"[CNN-Attn] 로드: {path} (Acc: {self.accuracy:.4f})")
            return True
        except Exception as e:
            logger.error(f"[CNN-Attn] 로드 실패: {e}")
            return False
