#!/usr/bin/env python3
# flaml_automl_longrun_v2.py (新しいラベルデータで再学習するバージョン)

from pathlib import Path
import pandas as pd
import joblib
from flaml import AutoML

# --- ユーザ設定 ---
BASE_DIR = Path(__file__).resolve().parent # スクリプトがあるフォルダを基準とする
# ★★★ 新しい学習データファイル名に変更 ★★★
TRAINING_DATA_FILE = BASE_DIR / "step2_time_features_new_label.csv"

# ★★★ モデルが期待する特徴量のリスト (check_model_features.py の出力から正確に記述) ★★★
# (label列と、もしあればbar_time列はここには含めないでください)
EXPECTED_MODEL_FEATURES = [
    "entry_price", "exit_price", "ema_fast", "ema_mid", "ema_slow", "K", "D",
    "open", "high", "low", "close", "ATR_14", "KD_angle", "body", "upper_wick",
    "lower_wick", "hour", "weekday", # 'hour'と'weekday'の元々の数値特徴量も含む
    "hour_0", "hour_1", "hour_2", "hour_3", "hour_4", "hour_5", "hour_6", 
    "hour_7", "hour_8", "hour_9", "hour_10", "hour_11", "hour_12", "hour_13",
    "hour_14", "hour_15", "hour_16", "hour_17", "hour_18", "hour_19", 
    "hour_20", "hour_21", "hour_22", "hour_23",
    "weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4"
]
# --- ここまで ---


def train_flaml_model():
    print(f"[INFO] 学習データファイル: {TRAINING_DATA_FILE.name}")
    try:
        df = pd.read_csv(TRAINING_DATA_FILE)
        print(f"[INFO] 学習データをロードしました。 ({len(df)}行 x {len(df.columns)}列)")
    except FileNotFoundError:
        print(f"[ERR] 学習データファイルが見つかりません: {TRAINING_DATA_FILE}")
        return
    except Exception as e:
        print(f"[ERR] 学習データのロード中にエラー: {e}")
        return

    # 特徴量 X と ラベル y の準備
    if 'label' not in df.columns:
        print("[ERR] 学習データに 'label' カラムが見つかりません。")
        return
    
    # EXPECTED_MODEL_FEATURES に含まれるカラムだけをXとして使う
    missing_features_in_csv = [col for col in EXPECTED_MODEL_FEATURES if col not in df.columns]
    if missing_features_in_csv:
        print(f"[ERR] 学習データに必要な特徴量が不足しています: {missing_features_in_csv}")
        print(f"       TRAINING_DATA_FILE ({TRAINING_DATA_FILE.name}) のカラムを確認してください。")
        return
        
    try:
        X = df[EXPECTED_MODEL_FEATURES]
        y = df["label"]
        print(f"[INFO] 特徴量X ({X.shape[0]}行 x {X.shape[1]}列), ラベルy を準備しました。")
    except KeyError as e:
        print(f"[ERR] 特徴量リストに誤りがあるか、学習データに必要なカラムがありません: {e}")
        return
    except Exception as e:
        print(f"[ERR] X, y の準備中に予期せぬエラー: {e}")
        return

    # AutoML インスタンスの作成と設定
    # (log_verbosityはAutoMLのコンストラクタで設定するのが一般的ですが、fitのsettingsでも渡せます)
    automl = AutoML() 

    settings = {
        "time_budget": 15 * 3600,  # 15時間 (パンさんの以前の設定)
        "metric": "f1",            # 評価指標
        "task": "classification",
        "eval_method": "cv",       # クロスバリデーション
        "n_splits": 5,             # CVの分割数 (eval_methodが"cv"の場合)
        "split_type": "time",  # ★★★ これを追加 ★★★ (時系列データを考慮した分割)
        "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"], # 試したい学習器
        "log_file_name": "flaml_retrain_v2.log",  # ★ 新しいログファイル名
        "log_type": "all",         # ★ リーダーボード用に "all" を推奨
        "n_jobs": 2,               # 使用CPUコア数 (パンさんの以前の設定)
        # "log_verbosity": 3,      # AutoML()の引数で設定していれば不要な場合も
                                 # fitに渡すと上書きされるか、あるいは両方設定しても問題ないことが多い
    }
    # もしAutoMLの初期化で log_verbosity を設定したい場合:
    # automl = AutoML(log_verbosity=3) # この行を settings の上の AutoML() の代わりに使う

    print(f"[INFO] FLAMLによるモデルの再学習を開始します...")
    print(f"       時間上限: {settings['time_budget']/3600:.1f}時間")
    print(f"       使用コア数上限: {settings['n_jobs']}")
    print(f"       評価指標: {settings['metric']}")
    
    automl.fit(X_train=X, y_train=y, **settings)
    print("[INFO] モデルの再学習が完了しました。")

    # 学習済みモデルの保存
    NEW_MODEL_FILENAME = "best_flaml_model_v2.pkl" # ★ 新しいモデルファイル名
    output_model_path = BASE_DIR / NEW_MODEL_FILENAME
    try:
        joblib.dump(automl, output_model_path)
        print(f"✓ 新しい学習済みモデルを '{output_model_path.name}' に保存しました。")
    except Exception as e:
        print(f"エラー: モデルの保存に失敗しました: {e}")

    # 最良モデルの情報表示
    print("\n--- FLAMLが見つけた最良モデル (再学習後) ---")
    if hasattr(automl, 'best_estimator') and automl.best_estimator is not None:
        print(f"最良の学習器: {automl.best_estimator}")
        print(f"最良のハイパーパラメータ: {automl.best_config}")
        print(f"最良のスコア (1 - {settings['metric']}): {automl.best_loss}")
        if settings['metric'] == 'f1':
            print(f"  (つまりF1スコアは約: {1.0 - automl.best_loss:.4f})")
        # 他のメトリックの場合の表示も追加可能
    else:
        print("最良モデルの情報が見つかりませんでした。学習時間や設定を確認してください。")


    # Leaderboard 情報の取得と表示/保存
    if hasattr(automl, 'training_log') and automl.training_log is not None:
        print("\n--- Leaderboard (上位5件) ---")
        leaderboard_df = pd.DataFrame(automl.training_log)
        
        if not leaderboard_df.empty and 'val_loss' in leaderboard_df.columns:
            sorted_leaderboard = leaderboard_df.sort_values('val_loss')
            print(sorted_leaderboard[['learner', 'val_loss', 'train_time', 'iter_per_learner']].head().to_string(index=False))
            
            leaderboard_csv_path = BASE_DIR / "flaml_leaderboard_v2.csv" # ★ 新しいLeaderboardファイル名
            try:
                sorted_leaderboard.to_csv(leaderboard_csv_path, index=False)
                print(f"\n✓ Leaderboardを '{leaderboard_csv_path.name}' に保存しました。")
            except Exception as e:
                print(f"エラー: LeaderboardのCSV保存に失敗: {e}")
        else:
            print("Leaderboard情報がtraining_logから取得できませんでした (空または必要な列なし)。")
    else:
        print("\nLeaderboard情報 (automl.training_log) は利用できません。settingsのlog_type='all'を確認してください。")

if __name__ == "__main__":
    train_flaml_model()