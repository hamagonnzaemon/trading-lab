# -*- coding: utf-8 -*-
"""
ファイル名: step2_time_features.py

ステップ2: 時刻情報の周期変換
  - 時間帯（1時間ごと）のワンホットダミー
  - 曜日（月〜日）のワンホットダミー
  - 結果を step2_time_features.csv として出力

使用方法: python step2_time_features.py
/usr/bin/python3 /Users/hamadakou/Desktop/trade_log/step2_time_features.py
上記ターミナルで実行
"""
import os
import pandas as pd

# スクリプト自身のディレクトリを基準にファイルパスを設定
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
input_file  = os.path.join(BASE_DIR, 'step1_features.csv')
output_file = os.path.join(BASE_DIR, 'step2_time_features.csv')

# 1. データ読み込み
df = pd.read_csv(input_file, parse_dates=['bar_time', 'entry_time', 'exit_time'])

# 2. 時間帯ダミー (hour_0 ... hour_23)
df['hour'] = df['bar_time'].dt.hour
hour_dummies = pd.get_dummies(df['hour'], prefix='hour')

# 3. 曜日ダミー (weekday_0=月曜 ... weekday_6=日曜)
df['weekday'] = df['bar_time'].dt.weekday
weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')

# 4. ダミー列を結合
df = pd.concat([df, hour_dummies, weekday_dummies], axis=1)

# 5. 中間列を削除（必要に応じて）
# df.drop(columns=['hour', 'weekday'], inplace=True)

# 6. 結果を CSV に出力
df.to_csv(output_file, index=False)
print(f"時間特徴量追加完了: {output_file}")
