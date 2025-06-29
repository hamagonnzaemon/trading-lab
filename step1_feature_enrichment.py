# -*- coding: utf-8 -*-
"""
ファイル名: step1_feature_enrichment.py
バージョン: v1.1.0 (2025-05-17)
   - v1.0.0: オリジナルバージョン
   - v1.1.0: NaN処理を変更。dropna()を削除し、主要テクニカル指標のNaN/infを0でフィル。
             (EAとのデータ整合性向上のため)

ステップ1: 機械学習用データの拡充
  - ATR、ローソク足の実体・ヒゲ長さ
  - %K-%D 角度の追加
  - 結果を step1_features.csv として出力

使用方法: python step1_feature_enrichment.py
"""
import os
import pandas as pd
import numpy as np

# スクリプト自身のディレクトリを基準にファイルパスを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 必要な CSV ファイルパスを設定
# (元ファイルのパスをそのまま使用します。環境に合わせて変更してください)
trades_file = os.path.join(BASE_DIR, 'trades_ml_roadmap.csv')
bars_file   = os.path.join(BASE_DIR, 'GOLD_M5_202101040100_202504302355.csv')
output_file = os.path.join(BASE_DIR, 'step1_features.csv')


# 2. CSV ファイルの存在チェック
if not os.path.exists(trades_file):
    raise FileNotFoundError(f"ファイルが見つかりません: {trades_file}")
if not os.path.exists(bars_file):
    raise FileNotFoundError(f"ファイルが見つかりません: {bars_file}")

# 3. trades_ml_roadmap.csv を読み込み（bar_time を日時型に変換）
trades = pd.read_csv(trades_file, parse_dates=['bar_time'])

# 4. ゴールド 5分足 CSV を読み込み（空白区切り）
#    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
bars = pd.read_csv(bars_file, delim_whitespace=True)

# 5. カラム名を統一（<DATE>,<TIME> 等の<>を除去して小文字化）
bars.columns = [col.strip('<>').lower() for col in bars.columns]
# 必要カラムが揃っているかチェック
required_bars_cols = ['date', 'time', 'open', 'high', 'low', 'close']
for col in required_bars_cols:
    if col not in bars.columns:
        raise KeyError(f"バーデータに必要カラムが見つかりません: {col}")

# 6. 日付と時刻を結合して bar_time カラムを作成
bars['bar_time'] = pd.to_datetime(bars['date'] + ' ' + bars['time'], format='%Y.%m.%d %H:%M:%S')

# 7. 不要な列を削除（解析に不要なボリューム等）し、必要な列を選択
bars = bars[['bar_time', 'open', 'high', 'low', 'close']] # 'tickvol' なども必要なら追加

# 8. trades とマージ
data = pd.merge(trades, bars, on='bar_time', how='left')

# マージ後の欠損チェック (OHLC価格データに欠損があると計算に影響するため)
missing_ohlc = data[['open','high','low','close']].isna().sum().sum()
if missing_ohlc > 0:
    print(f"警告: マージ後のOHLCデータに {missing_ohlc} 件の欠損があります。欠損行を削除または補間してください。")
    # ここで処理を停止するか、欠損行を削除するなどの対応が必要な場合があります。
    # 例: data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    # 今回は、ひとまず続行します。

# --- テクニカル指標の計算 ---

# 9. ATR (Average True Range) の計算
#    TR = Max(High - Low, Abs(High - PreviousClose), Abs(Low - PreviousClose))
#    ATR = SMA(TR, N)  <- EAの計算方法 (Rolling mean of TR)
tr_h_l = data['high'] - data['low']
tr_h_pc = (data['high'] - data['close'].shift()).abs()
tr_l_pc = (data['low']  - data['close'].shift()).abs()
tr = pd.concat([tr_h_l, tr_h_pc, tr_l_pc], axis=1).max(axis=1)
data['ATR_14'] = tr.rolling(window=14).mean().fillna(0) # 修正: NaNを0でフィル

# 10. 移動平均線 (EMA)
#     EA側の期間: fast=20, mid=50, slow=70
data['ema_fast'] = data['close'].ewm(span=20, adjust=False).mean()
data['ema_mid']  = data['close'].ewm(span=50, adjust=False).mean()
data['ema_slow'] = data['close'].ewm(span=70, adjust=False).mean()
# EMAは比較的NaNが発生しにくいが、入力データ先頭がNaNなら伝播する。
# 必要に応じて .fillna(0) も検討。今回はEAとの誤差が小さかったため変更なし。

# 11. ストキャスティクス (%K, %D)
#     EAパラメータ: KPeriod=15, Slowing=3, DPeriod=9, Method=SMA, PriceField=STO_CLOSECLOSE
k_period = 15
slowing_period = 3 # Slow %K の計算に使う期間 (EAのSlowingに対応)
d_period = 9     # Slow %D の計算に使う期間 (EAのDPeriodに対応)

# STO_CLOSECLOSE の再現:
# Raw %K (Fast %K) = (Current Close - Lowest Close_k_period) / (Highest Close_k_period - Lowest Close_k_period) * 100
rolling_close_min_k = data['close'].rolling(window=k_period).min()
rolling_close_max_k = data['close'].rolling(window=k_period).max()

# 0除算対策とNaN/inf処理
numerator = data['close'] - rolling_close_min_k
denominator = rolling_close_max_k - rolling_close_min_k
fast_k = (numerator / denominator) * 100
fast_k.replace([np.inf, -np.inf], np.nan, inplace=True) # inf を NaN に置換
fast_k = fast_k.fillna(0)                               # NaN を 0 でフィル (修正)

# Slow %K = SMA(Fast %K, slowing_period)
data['K'] = fast_k.rolling(window=slowing_period).mean().fillna(0) # 修正: NaNを0でフィル

# Slow %D = SMA(Slow %K, d_period)
data['D'] = data['K'].rolling(window=d_period).mean().fillna(0)    # 修正: NaNを0でフィル

# 12. %K-%D 角度 (簡易版, -90～+90度)
#     EA側は atan(K-D) * 180 / PI
#     KやDが0フィルされた影響で、角度も0になる箇所が増える可能性あり
data['KD_angle'] = np.arctan(data['K'] - data['D']) * 180 / np.pi
data['KD_angle'] = data['KD_angle'].fillna(0) # 修正: NaNを0でフィル

# 13. ローソク足の実体とヒゲの長さ (これらのNaN処理は任意)
data['body'] = (data['close'] - data['open']).abs()
data['upper_wick'] = data['high'] - data.apply(lambda x: max(x['open'], x['close']), axis=1)
data['lower_wick'] = data.apply(lambda x: min(x['open'], x['close']), axis=1) - data['low']
# data['body'] = data['body'].fillna(0) # 必要なら追加
# data['upper_wick'] = data['upper_wick'].fillna(0) # 必要なら追加
# data['lower_wick'] = data['lower_wick'].fillna(0) # 必要なら追加


# 14. 不要な行を削除 (NaN が含まれる期間初めの行など) -> この処理は削除
# data.dropna(inplace=True) # 削除またはコメントアウト

# 15. 結果をCSVファイルに出力
data.to_csv(output_file, index=False, float_format='%.5f')

print(f"処理完了: 特徴量データを {output_file} に保存しました。")
print(f"データ行数: {len(data)}")