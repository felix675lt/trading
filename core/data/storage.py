"""SQLite 기반 데이터 저장소"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

DB_PATH = Path("data/autotrader.db")


class Storage:
    """SQLite 저장소 - 캔들, 거래, 시그널, 모델 성능 기록"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                exchange TEXT,
                symbol TEXT,
                timeframe TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (exchange, symbol, timeframe, timestamp)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                exchange TEXT,
                symbol TEXT,
                side TEXT,
                price REAL,
                amount REAL,
                pnl REAL,
                fee REAL,
                strategy TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                model TEXT,
                signal REAL,
                confidence REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                accuracy REAL,
                sharpe REAL,
                win_rate REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TEXT PRIMARY KEY,
                equity REAL,
                drawdown REAL,
                positions TEXT
            );
        """)
        self.conn.commit()

    def save_candles(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame):
        records = []
        for ts, row in df.iterrows():
            records.append((
                exchange, symbol, timeframe, str(ts),
                row["open"], row["high"], row["low"], row["close"], row["volume"],
            ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", records
        )
        self.conn.commit()

    def load_candles(
        self, exchange: str, symbol: str, timeframe: str, limit: int = 5000
    ) -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE exchange=? AND symbol=? AND timeframe=?
            ORDER BY timestamp DESC LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=(exchange, symbol, timeframe, limit))
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df.sort_index()

    def save_trade(self, trade: dict):
        self.conn.execute(
            "INSERT INTO trades (timestamp, exchange, symbol, side, price, amount, pnl, fee, strategy, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                trade.get("timestamp", str(datetime.utcnow())),
                trade["exchange"], trade["symbol"], trade["side"],
                trade["price"], trade["amount"],
                trade.get("pnl", 0), trade.get("fee", 0),
                trade.get("strategy", ""), json.dumps(trade.get("metadata", {})),
            ),
        )
        self.conn.commit()

    def get_trades(self, limit: int = 100) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def save_equity(self, equity: float, drawdown: float, positions: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO equity_curve VALUES (?,?,?,?)",
            (str(datetime.utcnow()), equity, drawdown, json.dumps(positions)),
        )
        self.conn.commit()

    def save_model_performance(self, model_name: str, accuracy: float, sharpe: float, win_rate: float):
        self.conn.execute(
            "INSERT INTO model_performance (timestamp, model_name, accuracy, sharpe, win_rate, metadata) "
            "VALUES (?,?,?,?,?,?)",
            (str(datetime.utcnow()), model_name, accuracy, sharpe, win_rate, "{}"),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
