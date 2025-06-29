import pandas as pd
import numpy as np
import os
import warnings

"""
ML Dataset Preparation Script  v1.3.2
-------------------------------------
* パスを Desktop/trade_log に固定しない。`DIRECT_PATH` を編集して使う
* MetaTrader ヒストリカル CSV (tab 区切り) とエントリーログ (semicolon 区切り) を結合
* 追加特徴量: ローソク実体・ヒゲ・ATR14
* ラベル列 `y` は label → {'TP':1,'SL':0} で Int8(nullable) へ変換
  - ラベルが欠損していても落ちないように nullable 型を使う
* FutureWarning 対策: `dt.floor("5min")`
"""

# === ① ファイルパス設定 ==============================================
DIRECT_PATH = "/Users/hamadakou/Desktop/trade_log"  # ←必要に応じて変更
if not os.path.isdir(DIRECT_PATH):
    raise FileNotFoundError(f"指定されたログディレクトリが見つかりません: {DIRECT_PATH}")

ENTRY_CSV = os.path.join(DIRECT_PATH, "entry_log_all.csv")
BAR_CSV   = os.path.join(DIRECT_PATH, "GOLD_M5_202101040100_202504302355.csv")

# === ② ファイル存在確認 ===============================================
for path in (ENTRY_CSV, BAR_CSV):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

# === ③ CSV 読込 ======================================================
entry = pd.read_csv(ENTRY_CSV, sep=';')                      # EA ログ
bars  = pd.read_csv(BAR_CSV,   sep='\t', engine='python')    # MT5 5分足ヒストリ

# ── 時刻を datetime64 へ変換 ------------------------------------------
bars['datetime'] = pd.to_datetime(
    bars['<DATE>'].str.strip() + ' ' + bars['<TIME>'].str.strip(),
    format='%Y.%m.%d %H:%M', errors='coerce'
)

# 不要列整理 & リネーム
bars.rename(columns={
    '<OPEN>':'open','<HIGH>':'high','<LOW>':'low','<CLOSE>':'close'
}, inplace=True)

# === ④ バー側追加特徴量 ==============================================
bars['body']       = bars['close'] - bars['open']
bars['range']      = bars['high']  - bars['low']
bars['upper_wick'] = bars['high']  - bars[['open','close']].max(axis=1)
bars['lower_wick'] = bars[['open','close']].min(axis=1) - bars['low']

# ATR14 (EMA)
tr = np.maximum.reduce([
    bars['high'] - bars['low'],
    (bars['high'] - bars['close'].shift()).abs(),
    (bars['low']  - bars['close'].shift()).abs()
])
bars['ATR14'] = pd.Series(tr, index=bars.index).ewm(span=14, adjust=False).mean()

# === ⑤ エントリー & バー結合 =========================================
entry['datetime'] = pd.to_datetime(entry['datetime'], errors='coerce')
entry.sort_values('datetime', inplace=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    entry['bar_time'] = entry['datetime'].dt.floor('5min')   # 'T'→'min'

bars.set_index('datetime', inplace=True)

data = entry.merge(
    bars,
    left_on='bar_time', right_index=True,
    how='left', suffixes=('', '_bar')
)

# === ⑥ ラベル列 y (nullable) =========================================
if 'label' in data.columns:
    data['y'] = (data['label']
                 .replace({'TP':1, 'SL':0})
                 .astype('Int8'))
else:
    data['y'] = pd.Series([pd.NA]*len(data), dtype='Int8')

# === ⑦ 出力 ==========================================================
OUTPUT_CSV = os.path.join(DIRECT_PATH, 'ml_dataset.csv')
cols_keep = [
    # entry 側特徴
    'K','D','crossStrength','emaFast','emaMid','emaSlow',
    'emaGap_fast_slow','price_vs_emaFast',
    # bar 側特徴
    'body','range','upper_wick','lower_wick','ATR14',
    # label
    'y'
]

dataset = data[[c for c in cols_keep if c in data.columns]].dropna(subset=['K','D','emaFast'])
dataset.to_csv(OUTPUT_CSV, index=False)
print(f"✅ ml_dataset.csv を出力しました: {dataset.shape} -> {OUTPUT_CSV}")
