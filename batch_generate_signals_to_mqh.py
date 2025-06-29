#!/usr/bin/env python3
# batch_generate_signals_to_mqh.py
# 指定された期間の過去データに対してバッチでシグナルを生成し、
# MQL5 EAがインクルードできる .mqh ヘッダーファイルとして保存する。

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
from typing import Optional, List

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model_v2.pkl"
THRESHOLD_FILE = BASE_DIR / "best_threshold_v2.txt"
SOURCE_HISTORICAL_DATA_FILE = BASE_DIR / "realtime_bars_20210309_20250530.csv"
# MQL5 EAがインクルードするヘッダーファイル (MQL5/Include または EAと同じフォルダに配置)
OUTPUT_MQH_FILE = BASE_DIR / "signal_data.mqh" # Pythonスクリプトと同じ場所に一旦出力

# シグナルを生成する期間 (この期間でMT5バックテストを行う)
SIGNAL_PERIOD_START = "2025-05-01" # 例: 2025年5月
SIGNAL_PERIOD_END = "2025-05-31"
# --- ここまで ---

# --- モデルの列名取得関数 (変更なし) ---
def get_model_feature_names(model_path: Path) -> Optional[List[str]]:
    try:
        loaded_model = joblib.load(model_path)
        actual_estimator = None
        if hasattr(loaded_model, 'model') and loaded_model.model is not None: actual_estimator = loaded_model.model
        elif hasattr(loaded_model, 'fitted_estimator') and loaded_model.fitted_estimator is not None: actual_estimator = loaded_model.fitted_estimator
        else: actual_estimator = loaded_model
        if actual_estimator:
            if hasattr(actual_estimator, "feature_names_in_"): return list(map(str, actual_estimator.feature_names_in_))
            if hasattr(actual_estimator, "feature_name_"): return list(map(str, actual_estimator.feature_name_))
            if hasattr(actual_estimator, "booster_"):
                booster = actual_estimator.booster_
                if hasattr(booster, "feature_name") and callable(booster.feature_name): return list(map(str, booster.feature_name()))
                elif hasattr(booster, "feature_names") and isinstance(booster.feature_names, list): return list(map(str, booster.feature_names))
        if hasattr(loaded_model, "feature_names_in_"): return list(map(str, loaded_model.feature_names_in_))
        print("[WARN] モデルから特徴量名を特定の方法で見つけられませんでした。")
        return None
    except Exception as e: print(f"[ERR] モデルからの特徴量名取得中にエラー: {e}"); return None

# --- 特徴量生成関数 (変更なし) ---
def generate_features_for_batch(df_period_raw: pd.DataFrame, model_expected_cols: list[str]) -> Optional[pd.DataFrame]:
    if df_period_raw.empty: print("[WARN] generate_features_for_batch に渡されたDataFrameが空です。"); return None
    df_processed = df_period_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_processed["bar_time"]):
        try: df_processed["bar_time"] = pd.to_datetime(df_processed["bar_time"])
        except Exception as e: print(f"[ERR] bar_time をdatetimeに変換できませんでした: {e}"); return None
    df_processed["body"] = (df_processed["close"] - df_processed["open"]).abs()
    df_processed["upper_wick"]  = df_processed["high"] - np.maximum(df_processed["open"], df_processed["close"])
    df_processed["lower_wick"]  = np.minimum(df_processed["open"], df_processed["close"]) - df_processed["low"]
    df_processed["entry_price"] = df_processed["open"]
    df_processed["exit_price"] = df_processed["close"]
    df_processed["hour"] = df_processed["bar_time"].dt.hour
    df_processed["weekday"] = df_processed["bar_time"].dt.weekday
    for h_val in range(24):
        col_name = f"hour_{h_val}"
        if col_name in model_expected_cols: df_processed[col_name] = (df_processed["hour"] == h_val).astype(int)
    for wd_val in range(5): 
        col_name = f"weekday_{wd_val}"
        if col_name in model_expected_cols: df_processed[col_name] = (df_processed["weekday"] == wd_val).astype(int)
    for col in model_expected_cols:
        if col not in df_processed.columns:
            if col in df_period_raw.columns: df_processed[col] = df_period_raw[col]
            else: print(f"[WARN] 特徴量 '{col}' が元データにもなく生成もされません。0.0で埋めます。"); df_processed[col] = 0.0
    try: return df_processed[model_expected_cols]
    except KeyError as e: missing_cols = [col for col in model_expected_cols if col not in df_processed.columns]; print(f"[ERR] 特徴量の最終選択中にカラムが不足: {missing_cols}。エラー: {e}"); return None
    except Exception as e: print(f"[ERR] 特徴量生成の最終処理中に予期せぬエラー: {e}"); return None

# --- メイン処理 ---
def create_mqh_signal_file():
    print(f"[INFO] MQHシグナルファイル生成を開始します ({SIGNAL_PERIOD_START} ～ {SIGNAL_PERIOD_END})...")
    try:
        model = joblib.load(MODEL_FILE)
        threshold = float(Path(THRESHOLD_FILE).read_text().strip())
        print(f"[INFO] モデル ({MODEL_FILE.name}) と閾値 ({threshold:.4f}) をロードしました。")
    except Exception as e: print(f"[FATAL] モデルまたは閾値のロードに失敗: {e}"); return

    model_features = get_model_feature_names(MODEL_FILE)
    if not model_features: print("[FATAL] モデルから特徴量リストを取得できませんでした。"); return
    print(f"[INFO] モデルが要求する特徴量 ({len(model_features)}個) を取得しました。")

    try:
        df_full_history = pd.read_csv(SOURCE_HISTORICAL_DATA_FILE)
        df_full_history['bar_time'] = pd.to_datetime(df_full_history['bar_time'])
        # CSVの日時がナイーブな場合、UTCとして扱う
        if df_full_history['bar_time'].dt.tz is None:
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_localize('UTC', ambiguous='infer')
        else: # 既にタイムゾーン情報があればUTCに変換
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_convert('UTC')
        print(f"[INFO] 全履歴データ ({SOURCE_HISTORICAL_DATA_FILE.name}) をロードし、日時を処理しました。")
    except Exception as e: print(f"[FATAL] 履歴データのロードまたは日時処理に失敗: {e}"); return

    target_period_mask = (df_full_history['bar_time'] >= pd.to_datetime(SIGNAL_PERIOD_START, utc=True)) & \
                         (df_full_history['bar_time'] <= pd.to_datetime(SIGNAL_PERIOD_END, utc=True).replace(hour=23, minute=59, second=59))
    df_target_period = df_full_history[target_period_mask].copy()

    if df_target_period.empty: print(f"[ERR] 指定期間 ({SIGNAL_PERIOD_START} ～ {SIGNAL_PERIOD_END}) にデータがありません。"); return
    print(f"[INFO] 対象期間のデータ ({len(df_target_period)}行) を抽出しました。")

    print("[INFO] 対象期間データの特徴量を生成中...")
    # generate_features_for_batch は df_target_period のインデックスを保持して返すことを想定
    X_batch = generate_features_for_batch(df_target_period, model_features)

    if X_batch is None or X_batch.empty: print("[ERR] 特徴量生成に失敗しました。"); return
    X_batch.columns = X_batch.columns.astype(str)
    print(f"[INFO] 特徴量生成完了。予測に使用するデータ: {len(X_batch)}行")

    try:
        print("[INFO] モデルによる予測確率を計算中...")
        proba_batch = model.predict_proba(X_batch)[:, 1]
    except Exception as e: print(f"[ERR] 予測確率の計算中にエラー: {e}"); return

    # df_target_period から X_batch に対応する行だけを抽出 (dropnaなどでX_batchの行数が減った場合のため)
    df_target_period_for_signals = df_target_period.loc[X_batch.index]
    
    signals_for_mqh = []
    for i in range(len(df_target_period_for_signals)):
        bar_dt = df_target_period_for_signals['bar_time'].iloc[i]
        # MQL5のdatetimeリテラル形式 D'YYYY.MM.DD HH:MI:SS'
        mql_time_str = bar_dt.strftime("D'%Y.%m.%d %H:%M:%S'")
        signal_cmd = "BUY" if proba_batch[i] >= threshold else "NONE"
        signals_for_mqh.append(f"  {{{mql_time_str}, \"{signal_cmd}\"}}")

    # .mqhファイルの内容を生成
    mqh_content = "// signal_data.mqh - Generated by Python script\n\n"
    mqh_content += "#ifndef SIGNAL_DATA_MQH\n#define SIGNAL_DATA_MQH\n\n"
    mqh_content += "// Signal structure definition\n"
    mqh_content += "struct SignalInfoMqh { datetime time; string signal; };\n\n"
    mqh_content += "// Array of signals\n"
    mqh_content += "SignalInfoMqh G_SignalData[] = \n{\n"
    mqh_content += ",\n".join(signals_for_mqh) # 各エントリをカンマで結合
    mqh_content += "\n};\n\n"
    mqh_content += "#endif // SIGNAL_DATA_MQH\n"

    try:
        with open(OUTPUT_MQH_FILE, "w", encoding="utf-8") as f:
            f.write(mqh_content)
        print(f"\n✓ シグナルデータをMQHファイル '{OUTPUT_MQH_FILE.name}' に保存しました。({len(signals_for_mqh)}件)")
        print(f"  このファイルをEA (`.mq5`) と同じフォルダ、またはMQL5のIncludeフォルダにコピーしてください。")
    except Exception as e:
        print(f"[ERR] MQHファイルの書き出しに失敗: {e}")

if __name__ == "__main__":
    create_mqh_signal_file()