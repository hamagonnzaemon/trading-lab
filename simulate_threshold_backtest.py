#!/usr/bin/env python3
# simulate_threshold_backtest.py
#
# 使い方（ターミナル例）
#   python simulate_threshold_backtest.py \
#          --bars exported_bars.csv \
#          --model classweight_model.pkl \
#          --threshold 0.50 \
#          --hold_bars 12         # 何本先で決済するか（＝12本で約1時間保有）
#python3 simulate_threshold_backtest.py --bars exported_bars_ranged.csv --model classweight_model.pkl
#
# 出力 : 集計結果と 5 分足ごとの確率・シグナル・P/L を付与した CSV

#!/usr/bin/env python3
# simulate_threshold_backtest.py  v1.1  (2025-05-20)

import pandas as pd, joblib, numpy as np, argparse, sys
from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
LEARNED_COLS = [
    'ema_fast','ema_mid','ema_slow','K','D','open','high','low','close',
    'ATR_14','body_length','upper_wick','lower_wick','KD_angle',
    'hour','weekday',
    *[f"hour_{h}" for h in range(24)],
    *[f"weekday_{d}" for d in range(5)]
]
# ----------------------------------------------------------------------
def enrich_features(df):
    """ExportBars CSV に足りない列を作成して LEARNED_COLS を満たす"""
    if not pd.api.types.is_datetime64_any_dtype(df['bar_time']):
        df['bar_time'] = pd.to_datetime(df['bar_time'])

    # --- ローソク関連
    df['body_length'] = (df['close'] - df['open']).abs()
    df['upper_wick']  = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_wick']  = np.minimum(df['open'], df['close']) - df['low']

    # --- 時間・曜日
    df['hour']    = df['bar_time'].dt.hour
    df['weekday'] = df['bar_time'].dt.weekday

    for h in range(24):
        df[f'hour_{h}'] = (df['hour'] == h).astype(int)
    for wd in range(5):                  # 学習データは平日のみ
        df[f'weekday_{wd}'] = (df['weekday'] == wd).astype(int)

    # 余分な列はこの後 filter されるので放置で OK
    return df

# ----------------------------------------------------------------------
def backtest(bars_path, model_path, thr, hold_bars):
    df = pd.read_csv(bars_path, parse_dates=['bar_time'])
    df = enrich_features(df)

    miss = [c for c in LEARNED_COLS if c not in df.columns]
    if miss:
        print("[FATAL] まだ欠損列があります →", miss); sys.exit(1)

    model = joblib.load(model_path)
    X = df[LEARNED_COLS]
    prob = model.predict_proba(X)[:,1]
    df['proba'] = prob
    df['signal'] = (prob >= thr).astype(int)

    # --- エントリー＆ホールド単純検証 ---
    equity = 0
    trades = []
    for i,row in df.iterrows():
        if row['signal'] == 1:
            entry_price = row['close']
            exit_idx    = i + hold_bars
            if exit_idx >= len(df): break
            exit_price  = df.loc[exit_idx,'close']
            pnl = exit_price - entry_price      # ロング想定
            trades.append(pnl)
            equity += pnl

    win = sum(p>0 for p in trades)
    print(f"総トレード数 {len(trades)} / WinRate {win/len(trades):.2%} / "
          f"PF {(sum(p for p in trades if p>0) / abs(sum(p for p in trades if p<0))):.2f}"
          f" / 総損益 {equity:.2f}")

    # 予測付き CSV 出力
    out_file = bars_path.with_name("bars_with_proba.csv")
    df.to_csv(out_file, index=False)
    print("✅ 予測列付き CSV を保存:", out_file)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True, help="ExportBars CSV")
    ap.add_argument("--model", required=True, help="LightGBM pkl")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="BUY 判定に使う閾値 (def=0.50)")
    ap.add_argument("--hold_bars", type=int, default=12,
                    help="エントリー後保持するバー数 (def=12: M5×1h)")
    args = ap.parse_args()

    backtest(Path(args.bars), Path(args.model),
             args.threshold, args.hold_bars)

