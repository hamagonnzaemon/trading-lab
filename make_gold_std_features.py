# make_gold_std_features.py
import pandas as pd
from pathlib import Path

SRC   = Path("/Users/hamadakou/Desktop/trade_log/step2_time_features.csv")      # ← 今アップした学習用ファイル
DEST  = Path("/Users/hamadakou/Desktop/trade_log/gold_std_features.csv")        # 正解表の出力先
N_BARS = 576                                 # 48h / 5min ＝ 576 本

# 学習時と同じ列順
LEARNED_FEATURE_COLUMNS = [
    'ema_fast','ema_mid','ema_slow','K','D','open','high','low','close',
    'ATR_14','body_length','upper_wick','lower_wick','KD_angle',
    'hour','weekday',
    'hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7',
    'hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14',
    'hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21',
    'hour_22','hour_23',
    'weekday_0','weekday_1','weekday_2','weekday_3','weekday_4'
]

# ① CSV 読み込み（bar_time は必ず日付型で取り込む）
df = pd.read_csv(SRC, parse_dates=['bar_time'])

# ② 時間で 48h 切り出す **か** 行数で切り出す
use_time_window = False      # True にすると「最新時刻‑48h」のフィルタになる

if use_time_window:
    latest_ts = df['bar_time'].max()
    start_ts  = latest_ts - pd.Timedelta(hours=48)
    df_clip   = df[df['bar_time'] >= start_ts].copy()
else:
    df_clip   = df.tail(N_BARS).copy()

# ③ 列を LEARNED_FEATURE_COLUMNS 順に並べる
df_clip = df_clip[LEARNED_FEATURE_COLUMNS]

# ④ 保存
df_clip.to_csv(DEST, index=False)
print(f"saved {len(df_clip)} rows → {DEST}")
