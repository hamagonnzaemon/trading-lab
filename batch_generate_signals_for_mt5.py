#!/usr/bin/env python3
# batch_generate_signals_for_mt5.py
# 指定された期間の過去データに対してバッチでシグナルを生成し、MT5 EA用のCSVとして保存する。

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
# EAが出力したOHLCV + テクニカル指標が含まれる全期間データファイル
SOURCE_HISTORICAL_DATA_FILE = BASE_DIR / "realtime_bars_20210309_20250530.csv"
# MT5 EAが読み込むシグナルファイル (MQL5/Filesフォルダに配置すること)
OUTPUT_MT5_SIGNAL_FILE = BASE_DIR / "mt5_signals_for_backtest.csv"

# シグナルを生成する期間 (この期間でMT5バックテストを行う)
# 例えば2025年5月全体、または検証したい任意の期間
SIGNAL_PERIOD_START = "2025-05-01"
SIGNAL_PERIOD_END = "2025-05-31" # この日付まで含める
# --- ここまで ---

# --- モデルの列名取得関数 (他スクリプトから引用・調整) ---
def get_model_feature_names(model_path: Path) -> Optional[List[str]]:
    try:
        loaded_model = joblib.load(model_path)
        actual_estimator = None
        if hasattr(loaded_model, 'model') and loaded_model.model is not None:
            actual_estimator = loaded_model.model
        elif hasattr(loaded_model, 'fitted_estimator') and loaded_model.fitted_estimator is not None:
            actual_estimator = loaded_model.fitted_estimator
        else:
            actual_estimator = loaded_model
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

# --- 特徴量生成関数 (他スクリプトから引用・調整) ---
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
def generate_batch_signals():
    print(f"[INFO] バッチシグナル生成を開始します ({SIGNAL_PERIOD_START} ～ {SIGNAL_PERIOD_END})...")
    try:
        model = joblib.load(MODEL_FILE)
        threshold = float(Path(THRESHOLD_FILE).read_text().strip())
        print(f"[INFO] モデル ({MODEL_FILE.name}) と閾値 ({threshold}) をロードしました。")
    except Exception as e: print(f"[FATAL] モデルまたは閾値のロードに失敗: {e}"); return

    model_features = get_model_feature_names(MODEL_FILE)
    if not model_features: print("[FATAL] モデルから特徴量リストを取得できませんでした。"); return
    print(f"[INFO] モデルが要求する特徴量 ({len(model_features)}個) を取得しました。")

    try:
        df_full_history = pd.read_csv(SOURCE_HISTORICAL_DATA_FILE)
        df_full_history['bar_time'] = pd.to_datetime(df_full_history['bar_time'])
        if df_full_history['bar_time'].dt.tz is None:
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_localize('UTC', ambiguous='infer')
        else:
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_convert('UTC')
        print(f"[INFO] 全履歴データ ({SOURCE_HISTORICAL_DATA_FILE.name}) をロードし、日時を処理しました。")
    except Exception as e: print(f"[FATAL] 履歴データのロードまたは日時処理に失敗: {e}"); return

    # 指定された期間でデータをフィルタリング
    target_period_mask = (df_full_history['bar_time'] >= pd.to_datetime(SIGNAL_PERIOD_START, utc=True)) & \
                         (df_full_history['bar_time'] <= pd.to_datetime(SIGNAL_PERIOD_END, utc=True).replace(hour=23, minute=59, second=59))
    df_target_period = df_full_history[target_period_mask].copy()

    if df_target_period.empty: print(f"[ERR] 指定期間 ({SIGNAL_PERIOD_START} ～ {SIGNAL_PERIOD_END}) にデータがありません。"); return
    print(f"[INFO] 対象期間のデータ ({len(df_target_period)}行) を抽出しました。")

    print("[INFO] 対象期間データの特徴量を生成中...")
    X_batch = generate_features_for_batch(df_target_period, model_features)

    if X_batch is None or X_batch.empty: print("[ERR] 特徴量生成に失敗しました。"); return
    X_batch.columns = X_batch.columns.astype(str) # カラム名を文字列に統一
    print(f"[INFO] 特徴量生成完了。予測に使用するデータ: {len(X_batch)}行")

    try:
        print("[INFO] モデルによる予測確率を計算中...")
        proba_batch = model.predict_proba(X_batch)[:, 1]
    except Exception as e: print(f"[ERR] 予測確率の計算中にエラー: {e}"); return

    df_signals = pd.DataFrame({
        'timestamp_utc': df_target_period.loc[X_batch.index, 'bar_time'].dt.strftime('%Y-%m-%d %H:%M:%S'), # MT5で読みやすい形式
        'predicted_signal': np.where(proba_batch >= threshold, "BUY", "NONE")
        # 'probability': proba_batch # 必要なら確率も保存
    })
    
    # 予測結果がX_batchの行数と一致することを確認（generate_featuresでdropnaした場合を考慮）
    if len(df_signals) != len(proba_batch):
         print(f"[WARN] 予測結果の行数 ({len(proba_batch)}) が特徴量の行数 ({len(df_signals)}) と一致しません。インデックスを確認してください。")
         # ここでは単純にproba_batchの長さに合わせるが、実際はインデックスの整合性を確認すべき
         # df_signals = df_signals.iloc[:len(proba_batch)]
         # df_signals['predicted_signal'] = np.where(proba_batch >= threshold, "BUY", "NONE")

    try:
        df_signals.to_csv(OUTPUT_MT5_SIGNAL_FILE, index=False, header=True)
        print(f"\n✓ シグナル履歴を '{OUTPUT_MT5_SIGNAL_FILE.name}' に保存しました。({len(df_signals)}行)")
        print("   このファイルをMT5の MQL5/Files フォルダにコピーしてください。")
    except Exception as e:
        print(f"[ERR] シグナル履歴のCSV保存に失敗: {e}")

if __name__ == "__main__":
    generate_batch_signals()