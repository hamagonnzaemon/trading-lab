import pandas as pd

# フルパスを使って読み込み
bars = pd.read_csv(
    "/Users/hamadakou/Desktop/trade_log/GOLD_M5_202101040100_202504302355.csv",
    delim_whitespace=True, engine="python",
    names=["datetime","open","high","low","close","tickvol","vol","spread"],
    header=None, skiprows=1
)
print("バーCSV 列名:", bars.columns.tolist())

trades = pd.read_csv("/Users/hamadakou/Desktop/trade_log/generated_entry_log.csv")
print("トレードCSV 列名:", trades.columns.tolist())
