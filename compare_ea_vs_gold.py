#!/usr/bin/env python3
# compare_ea_vs_gold.py
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# 1. ファイルパス
# ─────────────────────────────────────────────
ea_csv_path   = Path("/Users/hamadakou/Desktop/trade_log/exported_bars_ranged.csv")         # EA 側
gold_csv_path = Path("/Users/hamadakou/Desktop/trade_log/step2_time_features.csv")   # Gold standard 側

# ─────────────────────────────────────────────
# 2. CSV 読み込み
# ─────────────────────────────────────────────
#  ※ bar_time 列の名前が違う場合は parse_dates の列名を合わせてください
ea_df   = pd.read_csv(ea_csv_path,   parse_dates=["bar_time"])
gold_df = pd.read_csv(gold_csv_path, parse_dates=["bar_time"])

# ─────────────────────────────────────────────
# 3. bar_time で内部結合
# ─────────────────────────────────────────────
merged = pd.merge(gold_df, ea_df, on="bar_time", suffixes=("_gold", "_ea"))
print("共通バー数:", len(merged))

# ─────────────────────────────────────────────
# 4. 共通する数値列だけ抽出
# ─────────────────────────────────────────────
num_cols_gold = [
    c for c in merged.columns
    if c.endswith("_gold") and pd.api.types.is_numeric_dtype(merged[c])
]
common_cols = [
    c[:-5]                       # "_gold" を削った元の列名
    for c in num_cols_gold
    if f"{c[:-5]}_ea" in merged.columns
]

if not common_cols:
    print("共通する数値列がありません。スクリプト終了。")
    exit()

# ─────────────────────────────────────────────
# 5. 列別の絶対誤差を計算
# ─────────────────────────────────────────────
abs_diff = {}
for col in common_cols:
    diff = (merged[f"{col}_ea"] - merged[f"{col}_gold"]).abs()
    abs_diff[col] = {"max": diff.max(), "mean": diff.mean()}

# ─────────────────────────────────────────────
# 6. 結果表示
# ─────────────────────────────────────────────
print("\n列別の絶対誤差（max / mean）")
for col, stats in abs_diff.items():
    print(f"{col:>12}: max={stats['max']:.6g}, mean={stats['mean']:.6g}")
