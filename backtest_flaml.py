#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_flaml_model.pkl を使った簡易バックテスト
- 予測確率 > THRESH で買いエントリー
- 同一バー終値で翌バー始値決済（超シンプル）
"""

import pandas as pd, joblib, numpy as np

# ----------------- パラメータ -----------------
THRESH = 0.60          # エントリー閾値 (要調整)
LOT    = 1.0           # 取引ロット（任意）
SPREAD = 0.0           # スプレッド pips (XM なら 1.5 など)

# ----------------- データ読み込み -------------
df_feat = pd.read_csv("step2_time_features.csv", parse_dates=["bar_time"])
price   = pd.read_csv("GOLD_M5_202101040100_202504302355.csv",
                      names=["time","open","high","low","close","volume"],
                      parse_dates=["time"])

# モデル読み込み
model = joblib.load("best_flaml_model.pkl")

# ----------------- 予測 ----------------------
X = df_feat.drop(columns=["label","entry_time","exit_time","bar_time"], errors="ignore")
proba = model.predict_proba(X)[:,1]
df_feat["prob"] = proba
df_feat["signal"] = (df_feat["prob"] > THRESH).astype(int)

# ----------------- バックテスト --------------
trades = []
in_position = False
entry_price = None

for idx, row in df_feat.iterrows():
    if not in_position and row["signal"] == 1:
        entry_price   = price.loc[idx, "close"] + SPREAD
        entry_time    = price.loc[idx, "time"]
        in_position   = True
    elif in_position:
        exit_price  = price.loc[idx, "open"] - SPREAD
        exit_time   = price.loc[idx, "time"]
        pnl         = (exit_price - entry_price) * LOT
        trades.append({"entry":entry_time,"exit":exit_time,
                       "entry_px":entry_price,"exit_px":exit_price,"pnl":pnl})
        in_position = False

bt = pd.DataFrame(trades)
bt["cum_pnl"] = bt["pnl"].cumsum()

# ----------------- 指標 ----------------------
total = bt["pnl"].sum()
win   = (bt["pnl"] > 0).sum()
loss  = (bt["pnl"] <=0).sum()
winrate = win / max(1, win+loss)
dd = (bt["cum_pnl"].cummax() - bt["cum_pnl"]).max()

print(f"取引回数   : {len(bt)}")
print(f"総損益     : {total:.2f}")
print(f"勝率        : {winrate:.2%}")
print(f"最大DD      : {dd:.2f}")

bt.to_csv("flaml_trades.csv", index=False)
print("✓ 取引履歴保存: flaml_trades.csv")
