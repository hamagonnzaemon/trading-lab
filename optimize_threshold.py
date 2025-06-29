#!/usr/bin/env python3
# optimize_threshold.py - 検証用データを使って最適な判断閾値を探すスクリプト

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Optional, List # List を明示的にインポート
import datetime as dt # generate_features_for_backtest で使うため

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model.pkl"
LABELED_VALIDATION_FILE = BASE_DIR / "validation_data_2024_labeled.csv" # ラベル付けした検証データ
NEW_THRESHOLD_FILE = BASE_DIR / "best_threshold.txt" # 最適な閾値をここに保存
# --- ここまで ---

# --- モデルの列名取得関数 (前回と同様) ---
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
            if hasattr(actual_estimator, "feature_names_in_"):
                return list(map(str, actual_estimator.feature_names_in_))
            if hasattr(actual_estimator, "feature_name_"):
                return list(map(str, actual_estimator.feature_name_))
            if hasattr(actual_estimator, "booster_"):
                booster = actual_estimator.booster_
                if hasattr(booster, "feature_name") and callable(booster.feature_name):
                    return list(map(str, booster.feature_name()))
                elif hasattr(booster, "feature_names") and isinstance(booster.feature_names, list):
                     return list(map(str, booster.feature_names))
        if hasattr(loaded_model, "feature_names_in_"):
            return list(map(str, loaded_model.feature_names_in_))
        print("[WARN] モデルから特徴量名を特定の方法で見つけられませんでした。")
        return None
    except Exception as e:
        print(f"[ERR] モデルからの特徴量名取得中にエラー: {e}")
        return None

# --- 特徴量生成関数 (backtest_oos.py と同様のもの) ---
def generate_features_for_backtest(df_period_raw: pd.DataFrame, model_expected_cols: list[str]) -> Optional[pd.DataFrame]:
    if df_period_raw.empty:
        print("[WARN] generate_features_for_backtest に渡されたDataFrameが空です。")
        return None
    
    df_processed = df_period_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_processed["bar_time"]):
        try:
            df_processed["bar_time"] = pd.to_datetime(df_processed["bar_time"])
        except Exception as e:
            print(f"[ERR] bar_time をdatetimeに変換できませんでした: {e}")
            return None
    
    df_processed["body"] = (df_processed["close"] - df_processed["open"]).abs()
    df_processed["upper_wick"]  = df_processed["high"] - np.maximum(df_processed["open"], df_processed["close"])
    df_processed["lower_wick"]  = np.minimum(df_processed["open"], df_processed["close"]) - df_processed["low"]
    df_processed["entry_price"] = df_processed["open"]
    df_processed["exit_price"] = df_processed["close"]
    
    df_processed["hour"] = df_processed["bar_time"].dt.hour
    df_processed["weekday"] = df_processed["bar_time"].dt.weekday

    for h_val in range(24):
        col_name = f"hour_{h_val}"
        if col_name in model_expected_cols:
            df_processed[col_name] = (df_processed["hour"] == h_val).astype(int)
    for wd_val in range(5): 
        col_name = f"weekday_{wd_val}"
        if col_name in model_expected_cols:
            df_processed[col_name] = (df_processed["weekday"] == wd_val).astype(int)
            
    for col in model_expected_cols:
        if col not in df_processed.columns:
            # print(f"[WARN] 期待される特徴量 '{col}' が生成されていません。元のデータにあるか確認します。")
            if col in df_period_raw.columns: # 元データにあればそれを使う
                 df_processed[col] = df_period_raw[col]
            else:
                 print(f"[WARN] 特徴量 '{col}' が元データにもなく生成もされません。0.0で埋めます。")
                 df_processed[col] = 0.0

    try:
        return df_processed[model_expected_cols]
    except KeyError as e:
        missing_cols = [col for col in model_expected_cols if col not in df_processed.columns]
        print(f"[ERR] 特徴量の最終選択中にカラムが不足しています: {missing_cols}。エラー: {e}")
        return None
    except Exception as e:
        print(f"[ERR] 特徴量生成の最終処理中に予期せぬエラー: {e}")
        return None

# --- 閾値最適化のメイン処理 ---
def optimize_threshold():
    print("[INFO] 閾値の最適化を開始します...")

    # モデルのロード
    try:
        model = joblib.load(MODEL_FILE)
        print(f"[INFO] モデル ({MODEL_FILE.name}) をロードしました。")
    except Exception as e:
        print(f"[FATAL] モデルのロードに失敗: {e}")
        return

    # モデルが期待する特徴量リストを取得
    model_features = get_model_feature_names(MODEL_FILE)
    if not model_features:
        print("[FATAL] モデルから特徴量リストを取得できませんでした。")
        return
    print(f"[INFO] モデルが要求する特徴量 ({len(model_features)}個) を取得しました。")

    # ラベル付けされた検証データのロード
    try:
        df_validation_labeled = pd.read_csv(LABELED_VALIDATION_FILE)
        # actual_outcome が NaN の行はここで再度除去 (create_validation_labels.pyでも除去済みだが念のため)
        df_validation_labeled.dropna(subset=['actual_outcome'], inplace=True)
        if df_validation_labeled.empty:
            print(f"[ERR] ラベル付け済み検証データ ({LABELED_VALIDATION_FILE.name}) が空か、'actual_outcome'が全てNaNです。")
            return
        print(f"[INFO] ラベル付け済み検証データ ({LABELED_VALIDATION_FILE.name}) をロードしました。{len(df_validation_labeled)}行。")
    except Exception as e:
        print(f"[FATAL] ラベル付け済み検証データのロードに失敗: {e}")
        return

    # 検証データから特徴量 X_validation と正解ラベル y_validation_true を作成
    print("[INFO] 検証データの特徴量を生成中...")
    X_validation = generate_features_for_backtest(df_validation_labeled.copy(), model_features)
    
    if X_validation is None or X_validation.empty:
        print("[ERR] 検証データの特徴量生成に失敗しました。処理を中止します。")
        return
    
    # generate_features_for_backtest が model_features の順序で返すことを想定
    # y_validation_true は X_validation とインデックスを合わせる必要がある
    # df_validation_labeled から actual_outcome を取得し、X_validation のインデックスに合わせる
    y_validation_true = df_validation_labeled.loc[X_validation.index, 'actual_outcome'].astype(int)
    
    if len(X_validation) != len(y_validation_true):
        print(f"[ERR] X_validation ({len(X_validation)}行) と y_validation_true ({len(y_validation_true)}行) の行数が一致しません。")
        return

    print(f"[INFO] 特徴量生成完了。検証に使用するデータ: {len(X_validation)}行")

    # カラム名を文字列に統一
    X_validation.columns = X_validation.columns.astype(str)

    # モデルで予測確率を取得
    try:
        print("[INFO] モデルによる予測確率を計算中...")
        y_proba_validation = model.predict_proba(X_validation)[:, 1] # クラス1の確率
    except Exception as e:
        print(f"[ERR] 検証データの予測確率計算中にエラー: {e}")
        return

    # 閾値を0.05から0.95まで0.01刻みで試す
    thresholds = np.linspace(0.05, 0.95, int((0.95-0.05)/0.01) + 1)
    
    best_f1_score = -1.0
    optimal_threshold = 0.5 # デフォルト
    best_precision = 0
    best_recall = 0
    
    print("\n--- 閾値ごとの評価 ---")
    print("Threshold | F1-score | Precision | Recall")
    print("------------------------------------------")

    for thr in thresholds:
        y_pred_validation = (y_proba_validation >= thr).astype(int)
        
        f1 = f1_score(y_validation_true, y_pred_validation, zero_division=0)
        precision = precision_score(y_validation_true, y_pred_validation, zero_division=0)
        recall = recall_score(y_validation_true, y_pred_validation, zero_division=0)
        
        print(f"{thr:9.2f} | {f1:8.4f} | {precision:9.4f} | {recall:6.4f}")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            optimal_threshold = thr
            best_precision = precision
            best_recall = recall

    print("------------------------------------------")
    print(f"\n検証データでF1スコアを最大化する最適な閾値: {optimal_threshold:.4f}")
    print(f"  その時のF1スコア  : {best_f1_score:.4f}")
    print(f"  その時の適合率    : {best_precision:.4f}")
    print(f"  その時の再現率    : {best_recall:.4f}")

    # 最適な閾値でのClassification Report
    print("\n--- 最適な閾値での Classification Report (検証データ) ---")
    y_pred_optimal = (y_proba_validation >= optimal_threshold).astype(int)
    print(classification_report(y_validation_true, y_pred_optimal, zero_division=0))

    # 最適な閾値をファイルに保存
    try:
        with open(NEW_THRESHOLD_FILE, "w") as f:
            f.write(str(optimal_threshold))
        print(f"\n✓ 最適な閾値 {optimal_threshold:.4f} を '{NEW_THRESHOLD_FILE.name}' に保存しました。")
    except Exception as e:
        print(f"[ERR] 最適な閾値のファイル保存中にエラー: {e}")

if __name__ == "__main__":
    optimize_threshold()