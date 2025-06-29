#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP3 : signal_history_labeled.csv から特徴量を生成し trainset.csv を出力
※ テクニカル設定：
      • EMA  : 20, 50, 70
      • STOCH: k=15, d=3, smooth_k=9
"""

from pathlib import Path
import pandas as pd
import pandas_ta as ta   # pip install pandas-ta

# ────────────── パス設定 ──────────────
BASE   = Path.home() / "Desktop" / "trade_log"
DATA_D = BASE / "data"
SRC    = DATA_D / "signal_history_labeled.csv"
DST    = DATA_D / "trainset.csv"
# ────────────── テクニカル設定 ────────
EMA_SET      = [20, 50, 70]        # 変更点①
STO_K, STO_D = 15, 3               # 変更点②
STO_SMOOTH   = 9                   # 変更点③
# ────────────────────────────────

# 1) ログ読み込み
df = pd.read_csv(SRC, parse_dates=["timestamp_utc"])

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],              # 文字列 → datetime
    utc=True,                         # 必ず UTC を付与
    errors="coerce"                   # 変換失敗は NaT
)
df.dropna(subset=["timestamp_utc"], inplace=True)

# 2) EMA
for length in EMA_SET:
    df.ta.ema(length=length, append=True)

# 3) ストキャスティクス %K / %D（smooth_k に 9 を指定）
# pandas_ta では 'smooth_k' 引数で %K 平滑化期間を指定
df.ta.stoch(high="high",
            low="low",
            close="close",
            k=STO_K,
            d=STO_D,
            smooth_k=STO_SMOOTH,
            append=True)

# 4) ATR（そのまま継承。必要なければ削除可）
df.ta.atr(length=14, append=True)

# 5) ローソク足形状
df["body_length"] = (df["close"] - df["open"]).abs()
df["upper_wick"]  = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"]  = df[["open", "close"]].min(axis=1) - df["low"]

# 6) 時間帯ダミー
df["ts_jst"]  = df["timestamp_utc"].dt.tz_convert("Asia/Tokyo")
df["hour"]    = df["ts_jst"].dt.hour
df["weekday"] = df["ts_jst"].dt.weekday
df = pd.get_dummies(df, columns=["hour", "weekday"], drop_first=False)

# 7) 不要列を整理
df.drop(columns=["actual_outcome", "ts_jst"], inplace=True, errors="ignore")

# 8) 欠損削除 & 保存
df.dropna(inplace=True)
df.to_csv(DST, index=False)
print(f"✅ 特徴量付きデータ {len(df):,} 行を書き出しました → {DST}")
