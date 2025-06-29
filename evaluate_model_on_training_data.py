# evaluate_model_on_training_data.py

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from typing import Optional, List # List を明示的にインポート

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model.pkl"
TRAINING_DATA_FILE = BASE_DIR / "step2_time_features.csv"
# --- ここまで ---

def get_model_feature_names(model_path: Path) -> Optional[List[str]]:
    """
    保存されたモデルオブジェクトから特徴量名のリストを取得します。
    (check_model_features.py から必要な部分を抜粋・調整)
    """
    try:
        loaded_model = joblib.load(model_path)
        
        # FLAML AutoMLオブジェクトの場合、実際のモデルは .model や .best_model_for_estimator にある
        # または .fitted_estimator (BaseEstimatorを継承したFLAMLの推定器ラッパー)
        actual_estimator = None
        if hasattr(loaded_model, 'model') and loaded_model.model is not None: 
            actual_estimator = loaded_model.model
        elif hasattr(loaded_model, 'fitted_estimator') and loaded_model.fitted_estimator is not None:
            actual_estimator = loaded_model.fitted_estimator
        else: # joblibで直接scikit-learn互換モデルを保存した場合など
            actual_estimator = loaded_model
        
        if actual_estimator:
            if hasattr(actual_estimator, "feature_names_in_"): # scikit-learn >= 1.0
                return list(map(str, actual_estimator.feature_names_in_)) # Ensure all are strings
            if hasattr(actual_estimator, "feature_name_"): # scikit-learn < 1.0
                return list(map(str, actual_estimator.feature_name_))
            # LightGBM/XGBoostのネイティブモデルの場合の追加チェック
            if hasattr(actual_estimator, "booster_"):
                booster = actual_estimator.booster_
                if hasattr(booster, "feature_name") and callable(booster.feature_name): # LightGBM
                    return list(map(str, booster.feature_name()))
                elif hasattr(booster, "feature_names") and isinstance(booster.feature_names, list): # XGBoost
                     return list(map(str, booster.feature_names))


        # AutoMLオブジェクトのトップレベルもフォールバックとしてチェック
        if hasattr(loaded_model, "feature_names_in_"):
            return list(map(str, loaded_model.feature_names_in_))
        
        print("[WARN] モデルオブジェクトから特徴量名を特定の方法で見つけられませんでした。")
        return None

    except Exception as e:
        print(f"[ERR] モデルからの特徴量名取得中にエラー: {e}")
        return None

def evaluate_model_on_training_data():
    print(f"学習済みモデル: {MODEL_FILE.name}")
    print(f"学習データ: {TRAINING_DATA_FILE.name}")

    try:
        model = joblib.load(MODEL_FILE)
        print("モデルのロード完了。")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_FILE}")
        return
    except Exception as e:
        print(f"エラー: モデルのロード中に問題が発生しました: {e}")
        return

    try:
        df_train = pd.read_csv(TRAINING_DATA_FILE)
        print(f"学習データのロード完了。{len(df_train)}行 x {len(df_train.columns)}列。")
    except FileNotFoundError:
        print(f"エラー: 学習データファイルが見つかりません: {TRAINING_DATA_FILE}")
        return
    except Exception as e:
        print(f"エラー: 学習データのロード中に問題が発生しました: {e}")
        return

    expected_features = get_model_feature_names(MODEL_FILE)
    
    if not expected_features:
        print("エラー: モデルから期待される特徴量リストを取得できませんでした。")
        print("もし 'step2_time_features.csv' に 'label' 列以外が全て特徴量であるなら、")
        print("手動で特徴量リストを指定する必要があるかもしれません。")
        if 'label' in df_train.columns:
            print("フォールバック試行: 'label'カラムを除いた全てを特徴量とみなします。")
            expected_features = [col for col in df_train.columns if col != 'label']
        else:
            print("学習データに 'label' カラムが見つからず、特徴量も特定できません。処理を中断します。")
            return
    
    print(f"モデルが期待する特徴量 ({len(expected_features)}個) を使用します。")

    if 'label' not in df_train.columns:
        print(f"エラー: 学習データ '{TRAINING_DATA_FILE.name}' に 'label' カラムが見つかりません。")
        return

    try:
        # 期待される特徴量だけを、期待される順序で選択（モデルが順序を記憶していれば）
        # もしget_model_feature_namesがNoneを返した場合、上記のフォールバックが使われる
        X_train = df_train[expected_features]
        y_train_true = df_train['label']
    except KeyError as e:
        print(f"エラー: 学習データから特徴量またはラベルを選択中にKeyErrorが発生しました: {e}")
        print(f"期待された特徴量: {expected_features}")
        print(f"CSVファイルに存在するカラム: {list(df_train.columns)}")
        return
    except Exception as e:
        print(f"エラー: X_train, y_train_true の準備中にエラー: {e}")
        return
        
    print("特徴量 (X_train) と正解ラベル (y_train_true) を準備しました。")

    # モデルに渡す直前に、X_trainのカラム名を確実に文字列型に変換
    X_train.columns = X_train.columns.astype(str)
    print("X_trainのカラム名を文字列型に変換しました。")

    print("\nモデルによる予測を実行中...")
    try:
        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)[:, 1] # クラス1に属する確率
        print("予測完了。")
    except Exception as e:
        print(f"エラー: モデル予測中に問題が発生しました: {e}")
        # エラー時にXの情報を少し表示してみる
        print("X_trainの最初の5行:")
        print(X_train.head())
        print("X_trainのカラム:")
        print(X_train.columns)
        return

    print("\n--- モデルの学習データに対する性能評価 ---")
    print("\n■ Classification Report:")
    # zero_division=0 は、適合率などが0になる場合に警告を出さず0として扱う設定
    print(classification_report(y_train_true, y_pred, zero_division=0))

    print("\n■ ROC AUC Score:")
    try:
        auc = roc_auc_score(y_train_true, y_proba)
        print(f"{auc:.4f}")
    except ValueError as e:
        print(f"AUC計算中にエラー: {e} (例: y_trueに一方のクラスしか含まれていない場合など)")
    except Exception as e:
        print(f"AUC計算中に予期せぬエラー: {e}")


    print("\n--- 結果の比較のための参考情報 ---")
    print("FLAMLが学習時に報告したクロスバリデーションでの最良F1スコア（目安）: 約 0.5167 (error = 0.4833)")
    print("上記 Classification Report の中の F1スコア（macro avg や weighted avg、またはクラス1のF1）と比較してみてください。")
    print("もし、この訓練データでのスコアがCVスコアよりも著しく高い場合は、過学習の可能性があります。")
    print("もし、この訓練データでのスコアもCVスコアも低い場合は、モデルの表現力不足や特徴量の不足などが考えられます。")

if __name__ == "__main__":
    evaluate_model_on_training_data()