# -*- coding: utf-8 -*-
"""
ファイル名: step6_generate_signals.py

ステップ6: 固定 TP/SL (30/20 pips) でシグナル CSV を生成
  1. 学習済み classweight_model.pkl と best_threshold.txt を読み込む
  2. step2_time_features.csv の全バーに対し勝ち確率 proba を推論
  3. 閾値以上なら signal=1, 未満なら 0
  4. entry_time と TP/SL (固定 30/20) を含む signals_fixed.csv を出力

使用方法:
    python step6_generate_signals.py
EA 側では signals_fixed.csv をポーリングし、signal==1 の行で
TP=30 pips, SL=20 pips 固定で OrderSend() を実行してください。
"""
import os
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 入力ファイル
features_file = os.path.join(BASE_DIR, 'step2_time_features.csv')
model_file    = os.path.join(BASE_DIR, 'classweight_model.pkl')
threshold_file= os.path.join(BASE_DIR, 'best_threshold.txt')
# 出力ファイル
signals_file  = os.path.join(BASE_DIR, 'signals_fixed.csv')

# 1. データとモデル読み込み
print('>> ファイル読み込み')
features = pd.read_csv(features_file, parse_dates=['bar_time'])
model = joblib.load(model_file)
threshold = float(Path(threshold_file).read_text())
print(f'   閾値 (F1最適): {threshold:.4f}')

# 2. 予測確率を計算
drop_cols = ['entry_time', 'bar_time', 'exit_time', 'label', 'entry_price', 'exit_price']
proba = model.predict_proba(features.drop(columns=drop_cols))[:, 1]

# 3. シグナル判定
actions = (proba >= threshold).astype(int)

# 4. CSV 出力
df_out = features[['bar_time']].copy()
df_out['signal'] = actions
df_out['tp_pips'] = 30  # 固定 TP 30 pips
df_out['sl_pips'] = 20  # 固定 SL 20 pips
df_out.rename(columns={'bar_time': 'entry_time'}, inplace=True)

df_out.to_csv(signals_file, index=False)
print(f'>> シグナルを {signals_file} に保存しました')
