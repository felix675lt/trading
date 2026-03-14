"""LSTM 기반 시계열 예측 모델"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
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
        """시퀀스 데이터 생성"""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            Xs.append(X[i:i + self.seq_length])
            ys.append(y[i + self.seq_length])
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame, feature_cols: list[str], label_col: str = "label",
              epochs: int = 50, batch_size: int = 64, lr: float = 0.001):
        """모델 학습"""
        self.feature_columns = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        # 정규화
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std

        X_seq, y_seq = self._create_sequences(X, y)
        split = int(len(X_seq) * 0.8)

        train_ds = TensorDataset(
            torch.FloatTensor(X_seq[:split]),
            torch.LongTensor(y_seq[:split]),
        )
        test_X = torch.FloatTensor(X_seq[split:]).to(self.device)
        test_y = torch.LongTensor(y_seq[split:]).to(self.device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.model = LSTMNetwork(input_size=len(feature_cols)).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 검증
            self.model.eval()
            with torch.no_grad():
                pred = self.model(test_X).argmax(dim=1)
                acc = (pred == test_y).float().mean().item()
                if acc > best_acc:
                    best_acc = acc

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")

        self.accuracy = best_acc
        logger.info(f"LSTM 학습 완료 - 최고 정확도: {self.accuracy:.4f}")
        return self.accuracy

    def predict(self, df: pd.DataFrame) -> dict:
        """예측"""
        if self.model is None or self.mean is None:
            return {"signal": 0.0, "confidence": 0.0, "direction": "neutral"}

        X = df[self.feature_columns].values[-self.seq_length:].astype(np.float32)
        X = (X - self.mean) / self.std
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
