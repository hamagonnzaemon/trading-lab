# check_model_features.py (修正版)

import joblib
from pathlib import Path
from typing import Optional # ★★★★★ 1. この行で Optional をインポート ★★★★★

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model.pkl"
# --- ここまで ---

# ★★★★★ 2. 関数の型ヒントを Optional[list[str]] に修正 ★★★★★
def get_feature_names_from_model(model_path: Path) -> Optional[list[str]]:
    """保存されたモデルオブジェクトから特徴量名のリストを取得する"""
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
                return list(actual_estimator.feature_names_in_)
            if hasattr(actual_estimator, "feature_name_"):
                return list(actual_estimator.feature_name_)
            if hasattr(actual_estimator, "booster_") and hasattr(actual_estimator.booster_, "feature_name"):
                return list(actual_estimator.booster_.feature_name())
        
        if hasattr(loaded_model, "feature_names_in_"):
            return list(loaded_model.feature_names_in_)
        
        return None

    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        return None
    except Exception as e:
        print(f"エラー: モデル読み込み中または特徴量取得中に問題が発生しました: {e}")
        return None

# --- メインの処理 ---
if __name__ == "__main__":
    print(f"モデルファイル '{MODEL_FILE.name}' から特徴量名を取得します...")
    
    feature_list = get_feature_names_from_model(MODEL_FILE)
    
    if feature_list:
        print("-" * 50)
        print(f"モデルが学習時に使用した特徴量リスト ({len(feature_list)}個):")
        print("-" * 50)
        for feature in feature_list:
            print(feature)
        print("-" * 50)
        print("このリストを元に realtime_signal_generator.py の generate_features 関数を修正してください。")
    else:
        print("モデルから特徴量リストを自動で取得できませんでした。")