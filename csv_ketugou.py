import pandas as pd
from pathlib import Path

# ① データ読み込み
bars = pd.read_csv(
    "/Users/hamadakou/Desktop/trade_log/GOLD_M5_202101040100_202504302355.csv",
    delim_whitespace=True, engine="python",
    names=["datetime","open","high","low","close","tickvol","vol","spread"],
    header=None, skiprows=1
)
bars["datetime"] = pd.to_datetime(bars["datetime"])
bars = bars.sort_values("datetime")

trades = pd.read_csv(
    "/Users/hamadakou/Desktop/trade_log/generated_entry_log.csv",
    parse_dates=["entry_time","exit_time"]
)
trades = trades.sort_values("entry_time")

# ② merge_asof で「直前のバー」を引当て
#    direction="backward" で、trades.entry_time <= bars.datetime の最も近い行をマッチ
merged = pd.merge_asof(
    trades,
    bars,
    left_on="entry_time",
    right_on="datetime",
    direction="backward",
    suffixes=("","_bar")
)

# ③ 不要列があれば削る、確認して保存
print("マージ後の null カウント：")
print(merged.isna().sum())
out = Path("/Users/hamadakou/Desktop/trade_log/trades_ml_merged_asof.csv")
merged.to_csv(out, index=False)
print(f"✅ 出力完了 → {out}")
