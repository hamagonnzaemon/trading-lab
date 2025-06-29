#!/usr/bin/env python3
# prepare_training_data_with_new_labels.py (bar_time列も保存するバージョン)
# EAが出力した指標を含む過去データに、新しいTP/SLルールでラベルを付け、
# さらに時間特徴量などを追加して、FLAML再学習用のデータセットを作成するスクリプト。

import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
from typing import Optional, List # Listを明示的にインポート

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
SOURCE_HISTORICAL_DATA_FILE = BASE_DIR / "realtime_bars_20210309_20250530.csv"
OUTPUT_TRAINING_DATA_FILE = BASE_DIR / "step2_time_features_new_label.csv"

TRAINING_START_DATE = "2021-01-04"
TRAINING_END_DATE   = "2025-04-30"

TP_USD_GAIN = 3.0
SL_USD_LOSS = 2.0
MAX_HOLDING_BARS = 12

EXPECTED_MODEL_FEATURES = [
    "entry_price", "exit_price", "ema_fast", "ema_mid", "ema_slow", "K", "D",
    "open", "high", "low", "close", "ATR_14", "KD_angle", "body", "upper_wick",
    "lower_wick", "hour", "weekday",
    *[f"hour_{h}" for h in range(24)],
    *[f"weekday_{d}" for d in range(5)]
]
# --- ここまで ---

def apply_tp_sl_labeling(df_period: pd.DataFrame) -> pd.Series:
    print(f"[INFO] ラベル付けを開始します。対象期間のバー数: {len(df_period)}")
    labels = []
    entry_prices_series = df_period['close']

    for i in range(len(df_period)):
        if i + MAX_HOLDING_BARS >= len(df_period):
            labels.append(np.nan)
            continue

        entry_price = entry_prices_series.iloc[i]
        tp_target_price = entry_price + TP_USD_GAIN
        sl_target_price = entry_price - SL_USD_LOSS
        
        current_label = np.nan 
        outcome_found = False

        for k in range(1, MAX_HOLDING_BARS + 1):
            future_bar_index = i + k
            future_high = df_period['high'].iloc[future_bar_index]
            future_low = df_period['low'].iloc[future_bar_index]

            if future_low <= sl_target_price:
                current_label = 0
                outcome_found = True
                break
            if future_high >= tp_target_price:
                current_label = 1
                outcome_found = True
                break
        
        if not outcome_found:
            exit_price_at_timeout = df_period['close'].iloc[i + MAX_HOLDING_BARS]
            current_label = 1 if exit_price_at_timeout > entry_price else 0
        
        labels.append(current_label)
        if (i + 1) % 5000 == 0:
            print(f"[INFO] {i + 1} / {len(df_period)} 件のラベル付け処理完了...")
            
    print("[INFO] ラベル付け処理が完了しました。")
    return pd.Series(labels, index=df_period.index)

def generate_final_features_and_label(df_labeled_input: pd.DataFrame, model_feature_list: List[str]) -> Optional[pd.DataFrame]:
    if df_labeled_input.empty:
        print("[WARN] generate_final_features_and_label に渡されたDataFrameが空です。")
        return None
    
    df_final = df_labeled_input.copy()

    if 'bar_time' not in df_final.columns:
        print("[ERR] 'bar_time' 列が df_labeled_input に存在しません。処理を中断します。")
        return None

    df_final["body"] = (df_final["close"] - df_final["open"]).abs()
    df_final["upper_wick"]  = df_final["high"] - np.maximum(df_final["open"], df_final["close"])
    df_final["lower_wick"]  = np.minimum(df_final["open"], df_final["close"]) - df_final["low"]
    if 'entry_price' not in df_final.columns:
        df_final["entry_price"] = df_final["open"]
    if 'exit_price' not in df_final.columns:
        df_final["exit_price"] = df_final["close"]
    
    df_final["hour"] = df_final["bar_time"].dt.hour
    df_final["weekday"] = df_final["bar_time"].dt.weekday

    for h_val in range(24):
        col_name = f"hour_{h_val}"
        if col_name in model_feature_list:
            df_final[col_name] = (df_final["hour"] == h_val).astype(int)
    for wd_val in range(5):
        col_name = f"weekday_{wd_val}"
        if col_name in model_feature_list:
            df_final[col_name] = (df_final["weekday"] == wd_val).astype(int)
            
    for col in model_feature_list:
        if col not in df_final.columns:
            print(f"[WARN] 最終特徴量リストに '{col}' がありません。NaNで埋めます。")
            df_final[col] = np.nan
            
    try:
        df_final.dropna(subset=['label'], inplace=True)
        # 特徴量にNaNが残っていても、FLAML側で処理されることが多いので、
        # ここでのdropnaはlabelだけに限定します。
        # df_final.dropna(subset=model_feature_list, inplace=True) 

        if df_final.empty:
            print("[ERR] 特徴量生成・NaN除去後、有効なデータが残りませんでした。")
            return None
        
        # ★★★ 保存するカラムリストに 'bar_time' を追加 ★★★
        final_columns_to_keep = ['bar_time'] + model_feature_list + ['label']
        
        # 実際に存在するカラムのみを選択する（安全のため）
        actual_cols_to_return = [col for col in final_columns_to_keep if col in df_final.columns]
        if len(actual_cols_to_return) != len(final_columns_to_keep):
            missing_for_save = [col for col in final_columns_to_keep if col not in actual_cols_to_return]
            print(f"[WARN] CSV保存時、一部の要求カラムが見つかりませんでした: {missing_for_save}")

        return df_final[actual_cols_to_return]
    except KeyError as e:
        missing = [col for col in final_columns_to_keep if col not in df_final.columns]
        print(f"[ERR] 最終カラム選択でKeyError: {missing} が不足しています。エラー: {e}")
        return None
    except Exception as e:
        print(f"[ERR] 最終的な特徴量セット準備中に予期せぬエラー: {e}")
        return None

if __name__ == "__main__":
    print(f"[INFO] 履歴データファイル '{SOURCE_HISTORICAL_DATA_FILE.name}' を読み込んでいます...")
    try:
        df_source = pd.read_csv(SOURCE_HISTORICAL_DATA_FILE)
    except FileNotFoundError:
        print(f"[ERR] 履歴データファイルが見つかりません: {SOURCE_HISTORICAL_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"[ERR] 履歴データファイルの読み込み中にエラー: {e}")
        exit()

    print("[INFO] 'bar_time' カラムを処理し、タイムゾーンをUTCに設定しています...")
    try:
        df_source['bar_time'] = pd.to_datetime(df_source['bar_time'])
        if df_source['bar_time'].dt.tz is None:
            df_source['bar_time'] = df_source['bar_time'].dt.tz_localize('UTC', ambiguous='infer')
        else:
            df_source['bar_time'] = df_source['bar_time'].dt.tz_convert('UTC')
    except Exception as e:
        print(f"[ERR] 'bar_time' カラムの処理中にエラー: {e}")
        exit()

    print(f"[INFO] 学習用期間 ({TRAINING_START_DATE} ～ {TRAINING_END_DATE}) のデータを抽出しています...")
    training_period_mask = (df_source['bar_time'] >= pd.to_datetime(TRAINING_START_DATE, utc=True)) & \
                           (df_source['bar_time'] <= pd.to_datetime(TRAINING_END_DATE, utc=True).replace(hour=23, minute=59, second=59))
    df_training_period = df_source[training_period_mask].copy()

    if df_training_period.empty:
        print(f"[ERR] 指定された学習期間にデータが見つかりませんでした。日付を確認してください。")
    else:
        print(f"[INFO] 学習用データとして {len(df_training_period)} 行を抽出しました。")
        
        df_training_period['label'] = apply_tp_sl_labeling(df_training_period)
        
        initial_rows = len(df_training_period)
        df_training_period.dropna(subset=['label'], inplace=True)
        if not df_training_period.empty: # dropna後に空でないことを確認
             df_training_period['label'] = df_training_period['label'].astype(int)
        rows_after_labeling = len(df_training_period)
        print(f"[INFO] ラベル付け後、有効なデータは {rows_after_labeling} 行です。(NaN除去: {initial_rows - rows_after_labeling}行)")

        if rows_after_labeling > 0:
            df_final_training_data = generate_final_features_and_label(df_training_period, EXPECTED_MODEL_FEATURES)

            if df_final_training_data is not None and not df_final_training_data.empty:
                try:
                    df_final_training_data.to_csv(OUTPUT_TRAINING_DATA_FILE, index=False, encoding='utf-8-sig')
                    print(f"\n✓ 新しいラベルと特徴量が付与された学習データを '{OUTPUT_TRAINING_DATA_FILE.name}' に保存しました。")
                    print(f"  保存されたデータ行数: {len(df_final_training_data)}")
                    print(f"  カラムリスト (最初の数個): {df_final_training_data.columns[:5].tolist()}...")
                    print(f"  成功(1)ラベルの数: {int(df_final_training_data['label'].sum())}")
                    print(f"  失敗(0)ラベルの数: {len(df_final_training_data) - int(df_final_training_data['label'].sum())}")
                except Exception as e:
                    print(f"[ERR] 最終学習データのCSV保存中にエラー: {e}")
            else:
                print("[ERR] 最終的な学習データの生成に失敗しました。")
        else:
            print("[ERR] ラベル付け後、有効なデータが残らなかったため、処理を終了します。")