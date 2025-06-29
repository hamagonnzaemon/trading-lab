#!/usr/bin/env python3
# optimize_threshold_v2.py (改良版)
# 検証用データを使って最適な判断閾値を、複数の指標を考慮して探すスクリプト

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Optional, List
import datetime as dt

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model_v2.pkl"
LABELED_VALIDATION_FILE = BASE_DIR / "validation_data_2024_labeled.csv"
NEW_THRESHOLD_FILE = BASE_DIR / "best_threshold_v2.txt" # 最適な閾値をここに保存

# 正解ラベル定義に基づく利確・損切り値 (パンさんの定義)
TP_USD_GAIN = 3.0  # 成功時の平均的利益と仮定 (利確幅)
SL_USD_LOSS = 2.0  # 失敗時の平均的損失と仮定 (損切り幅)
# --- ここまで ---

# --- モデルの列名取得関数 (変更なし) ---
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

# --- 特徴量生成関数 (変更なし) ---
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
            if col in df_period_raw.columns:
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

# --- 閾値最適化のメイン処理 (改良版) ---
def optimize_threshold_v2():
    print(f"[INFO] 新しいモデル ({MODEL_FILE.name}) の閾値最適化を開始します...")
    # ... (モデル、特徴量リスト、検証データのロード部分は前回と同様なので省略) ...
    try:
        model = joblib.load(MODEL_FILE)
        print(f"[INFO] モデル ({MODEL_FILE.name}) をロードしました。")
    except Exception as e:
        print(f"[FATAL] モデルのロードに失敗: {e}")
        return

    model_features = get_model_feature_names(MODEL_FILE)
    if not model_features:
        print("[FATAL] モデルから特徴量リストを取得できませんでした。")
        return
    print(f"[INFO] モデルが要求する特徴量 ({len(model_features)}個) を取得しました。")

    try:
        df_validation_labeled = pd.read_csv(LABELED_VALIDATION_FILE)
        df_validation_labeled.dropna(subset=['actual_outcome'], inplace=True)
        if df_validation_labeled.empty:
            print(f"[ERR] ラベル付け済み検証データ ({LABELED_VALIDATION_FILE.name}) が空か、'actual_outcome'が全てNaNです。")
            return
        print(f"[INFO] ラベル付け済み検証データ ({LABELED_VALIDATION_FILE.name}) をロードしました。{len(df_validation_labeled)}行。")
    except Exception as e:
        print(f"[FATAL] ラベル付け済み検証データのロードに失敗: {e}")
        return

    print("[INFO] 検証データの特徴量を生成中...")
    X_validation = generate_features_for_backtest(df_validation_labeled.copy(), model_features)
    if X_validation is None or X_validation.empty:
        print("[ERR] 検証データの特徴量生成に失敗しました。処理を中止します。")
        return
    y_validation_true = df_validation_labeled.loc[X_validation.index, 'actual_outcome'].astype(int)
    if len(X_validation) != len(y_validation_true):
        print(f"[ERR] X_validation ({len(X_validation)}行) と y_validation_true ({len(y_validation_true)}行) の行数が一致しません。")
        return
    print(f"[INFO] 特徴量生成完了。検証に使用するデータ: {len(X_validation)}行")
    X_validation.columns = X_validation.columns.astype(str)

    print("[INFO] モデルによる予測確率を計算中...")
    try:
        y_proba_validation = model.predict_proba(X_validation)[:, 1]
    except Exception as e:
        print(f"[ERR] 検証データの予測確率計算中にエラー: {e}")
        return

    thresholds = np.linspace(0.05, 0.95, int((0.95-0.05)/0.01) + 1) # 0.01刻み
    
    results_data = [] # 各閾値の結果を保存するリスト

    for thr in thresholds:
        y_pred_validation = (y_proba_validation >= thr).astype(int)
        
        num_total = len(y_validation_true)
        num_predicted_buys = np.sum(y_pred_validation) # BUYと予測された数
        
        f1_cls1 = f1_score(y_validation_true, y_pred_validation, pos_label=1, zero_division=0)
        precision_cls1 = precision_score(y_validation_true, y_pred_validation, pos_label=1, zero_division=0)
        recall_cls1 = recall_score(y_validation_true, y_pred_validation, pos_label=1, zero_division=0)
        
        # 期待利益の簡易計算 (BUYシグナルが出た場合)
        # 適合率 * 利確幅 - (1-適合率) * 損切り幅
        # 適合率が0（BUY予測が0回）の場合は、期待値も計算不能（またはマイナス無限大）なので注意
        expected_value_per_signal = 0.0
        if num_predicted_buys > 0: # BUYと予測した場合のみ期待値を計算
            expected_value_per_signal = (precision_cls1 * TP_USD_GAIN) - ((1 - precision_cls1) * SL_USD_LOSS)
        
        results_data.append({
            "Threshold": round(thr, 2),
            "F1_Class1": round(f1_cls1, 4),
            "Precision_Class1": round(precision_cls1, 4),
            "Recall_Class1": round(recall_cls1, 4),
            "Num_BUY_Signals": num_predicted_buys,
            "Expected_Value_per_Signal": round(expected_value_per_signal, 4) if num_predicted_buys > 0 else np.nan
        })

    results_df = pd.DataFrame(results_data)
    
    print("\n--- 閾値ごとの評価 (検証データ: 2024年) ---")
    print(results_df.to_string(index=False))
    print("-----------------------------------------------------------------------------------------")

    # F1スコア(クラス1)が最大の閾値
    best_f1_row = results_df.loc[results_df["F1_Class1"].idxmax()]
    optimal_threshold_f1 = best_f1_row["Threshold"]
    print(f"\n検証データで【F1スコア(クラス1)を最大化】する閾値: {optimal_threshold_f1:.4f}")
    print(f"  その時のF1スコア(クラス1)   : {best_f1_row['F1_Class1']:.4f}")
    print(f"  その時の適合率  (クラス1)   : {best_f1_row['Precision_Class1']:.4f}")
    print(f"  その時の再現率  (クラス1)   : {best_f1_row['Recall_Class1']:.4f}")
    print(f"  その時のBUYシグナル数        : {int(best_f1_row['Num_BUY_Signals'])}")
    print(f"  その時のシグナル毎期待値     : {best_f1_row['Expected_Value_per_Signal']:.4f}")


    # シグナル毎期待値がプラスで、かつ最大となる閾値
    positive_ev_df = results_df[results_df["Expected_Value_per_Signal"] > 0]
    if not positive_ev_df.empty:
        best_ev_row = positive_ev_df.loc[positive_ev_df["Expected_Value_per_Signal"].idxmax()]
        optimal_threshold_ev = best_ev_row["Threshold"]
        print(f"\n検証データで【シグナル毎期待値を最大化（かつプラス）】する閾値: {optimal_threshold_ev:.4f}")
        print(f"  その時のF1スコア(クラス1)   : {best_ev_row['F1_Class1']:.4f}")
        print(f"  その時の適合率  (クラス1)   : {best_ev_row['Precision_Class1']:.4f}")
        print(f"  その時の再現率  (クラス1)   : {best_ev_row['Recall_Class1']:.4f}")
        print(f"  その時のBUYシグナル数        : {int(best_ev_row['Num_BUY_Signals'])}")
        print(f"  その時のシグナル毎期待値     : {best_ev_row['Expected_Value_per_Signal']:.4f}")
    else:
        print("\n検証データでシグナル毎期待値がプラスになる閾値は見つかりませんでした。")
        optimal_threshold_ev = optimal_threshold_f1 # フォールバックとしてF1最大のものを採用

    # どの閾値を採用するかはパンさんの判断になります。
    # ここでは例として、F1スコアを最大化する閾値を保存します。
    # もし期待値ベースの閾値を使いたければ、optimal_threshold = optimal_threshold_ev としてください。
    final_chosen_threshold = optimal_threshold_f1 
    print(f"\n--- 最終的に選択する閾値 (F1最大化ベース): {final_chosen_threshold:.4f} ---")


    print("\n--- 選択した閾値での Classification Report (検証データ) ---")
    y_pred_chosen = (y_proba_validation >= final_chosen_threshold).astype(int)
    print(classification_report(y_validation_true, y_pred_chosen, target_names=['Class 0 (Fail)', 'Class 1 (Success)'], zero_division=0))

    try:
        with open(NEW_THRESHOLD_FILE, "w") as f:
            f.write(str(round(final_chosen_threshold, 4)))
        print(f"\n✓ 選択した閾値 {final_chosen_threshold:.4f} を '{NEW_THRESHOLD_FILE.name}' に保存しました。")
    except Exception as e:
        print(f"[ERR] 最適な閾値のファイル保存中にエラー: {e}")

if __name__ == "__main__":
    optimize_threshold_v2()