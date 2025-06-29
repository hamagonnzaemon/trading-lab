#!/usr/bin/env python3
# evaluate_model_v2_on_training_data.py
# 新しいラベルで再学習したモデル(v2)の、訓練データに対する性能を評価するスクリプト

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from typing import Optional, List # List を明示的にインポート
import datetime as dt # get_model_feature_names内でdtは使わないが、他のスクリプトとの整合性で残してもOK

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model_v2.pkl"  # ★★★ 新しいモデルファイル ★★★
TRAINING_DATA_FILE = BASE_DIR / "step2_time_features_new_label.csv" # ★★★ 新しい学習データ ★★★
# --- ここまで ---

def get_model_feature_names(model_path: Path) -> Optional[List[str]]:
    """
    保存されたモデルオブジェクトから特徴量名のリストを取得します。
    """
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
        print("[WARN] モデルオブジェクトから特徴量名を特定の方法で見つけられませんでした。")
        return None
    except Exception as e:
        print(f"[ERR] モデルからの特徴量名取得中にエラー: {e}")
        return None

def evaluate_new_model_on_its_training_data():
    print(f"評価対象モデル: {MODEL_FILE.name}")
    print(f"使用する学習データ: {TRAINING_DATA_FILE.name}")

    try:
        model = joblib.load(MODEL_FILE)
        print("[INFO] モデルのロード完了。")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_FILE}")
        return
    except Exception as e:
        print(f"エラー: モデルのロード中に問題が発生しました: {e}")
        return

    try:
        df_train = pd.read_csv(TRAINING_DATA_FILE)
        print(f"[INFO] 学習データをロードしました。{len(df_train)}行 x {len(df_train.columns)}列。")
    except FileNotFoundError:
        print(f"エラー: 学習データファイルが見つかりません: {TRAINING_DATA_FILE}")
        return
    except Exception as e:
        print(f"エラー: 学習データのロード中に問題が発生しました: {e}")
        return

    expected_features = get_model_feature_names(MODEL_FILE)
    
    if not expected_features:
        print("エラー: モデルから期待される特徴量リストを取得できませんでした。")
        if 'label' in df_train.columns and 'bar_time' in df_train.columns: # bar_timeも除外対象と仮定
            print("フォールバック試行: 'label'と'bar_time'カラムを除いた全てを特徴量とみなします。")
            expected_features = [col for col in df_train.columns if col not in ['label', 'bar_time']]
        elif 'label' in df_train.columns:
            print("フォールバック試行: 'label'カラムを除いた全てを特徴量とみなします。")
            expected_features = [col for col in df_train.columns if col != 'label']
        else:
            print("学習データに 'label' カラムが見つからず、特徴量も特定できません。処理を中断します。")
            return
        print(f"[WARN] フォールバックで取得した特徴量リストを使用します: {len(expected_features)}個")
    
    print(f"[INFO] モデルが期待する特徴量 ({len(expected_features)}個) を使用します。")

    if 'label' not in df_train.columns:
        print(f"エラー: 学習データ '{TRAINING_DATA_FILE.name}' に 'label' カラムが見つかりません。")
        return

    try:
        # df_trainに必要な特徴量が存在するか確認
        missing_in_df = [col for col in expected_features if col not in df_train.columns]
        if missing_in_df:
            print(f"エラー: 学習データにモデルが必要とする特徴量が不足しています: {missing_in_df}")
            return
            
        X_train = df_train[expected_features]
        y_train_true = df_train['label']
    except KeyError as e:
        print(f"エラー: 学習データから特徴量またはラベルを選択中にKeyErrorが発生しました: {e}")
        return
    except Exception as e:
        print(f"エラー: X_train, y_train_true の準備中にエラー: {e}")
        return
        
    print("[INFO] 特徴量 (X_train) と正解ラベル (y_train_true) を準備しました。")

    X_train.columns = X_train.columns.astype(str)
    print("[INFO] X_trainのカラム名を文字列型に変換しました。")

    print("\n[INFO] モデルによる予測を実行中...")
    try:
        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)[:, 1]
        print("[INFO] 予測完了。")
    except Exception as e:
        print(f"エラー: モデル予測中に問題が発生しました: {e}")
        return

    print("\n--- 新しいモデルの訓練データに対する性能評価 ---")
    print("\n■ Classification Report:")
    print(classification_report(y_train_true, y_pred, zero_division=0))

    print("\n■ ROC AUC Score:")
    try:
        auc = roc_auc_score(y_train_true, y_proba)
        print(f"{auc:.4f}")
    except ValueError as e:
        print(f"AUC計算中にエラー: {e}")
    except Exception as e:
        print(f"AUC計算中に予期せぬエラー: {e}")

    print("\n--- 結果の比較のための参考情報 ---")
    print("FLAMLが今回の再学習時に報告したクロスバリデーションでの最良F1スコア（目安）: 約 0.4685 (error = 0.5315)")
    print("上記 Classification Report の中の F1スコア（特にクラス1、またはmacro avg / weighted avg）と比較してください。")
    print("もし、この訓練データでのスコアがCVスコアよりも著しく高い場合は、過学習の可能性があります。")
    print("もし、この訓練データでのスコアもCVスコアも低い場合は、モデルの表現力不足や特徴量の不足などが考えられます。")

if __name__ == "__main__":
    evaluate_new_model_on_its_training_data()