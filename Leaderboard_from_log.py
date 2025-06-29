#!/usr/bin/env python3
# Leaderboard_from_log.py (JSONログ対応版)
# FLAMLのログファイル (flaml_longrun.log) からLeaderboard情報を抽出します。

import pandas as pd
import pathlib
import json # JSONをパースするためにインポート

# --- 設定 ---
LOG_FILE_PATH = pathlib.Path(__file__).resolve().parent / "flaml_longrun.log"
OUTPUT_CSV_PATH = pathlib.Path(__file__).resolve().parent / "flaml_leaderboard_from_log.csv"
# --- ここまで ---

def parse_flaml_log_json(log_path: pathlib.Path) -> pd.DataFrame:
    """
    FLAMLのJSON形式ログファイルをパースして、試行ごとの情報をDataFrameとして返します。
    """
    rows = []
    print(f"[INFO] ログファイル '{log_path.name}' を解析しています...")
    
    try:
        with log_path.open('r', encoding='utf-8') as f:
            for line_number, line_content in enumerate(f, 1):
                try:
                    # 各行が独立したJSONオブジェクトであると仮定
                    log_entry = json.loads(line_content)
                    
                    # 必要な情報を抽出
                    trial_info = {}
                    trial_info['iter_per_learner'] = log_entry.get('iter_per_learner', log_entry.get('record_id', line_number -1)) # イテレーション番号
                    trial_info['learner'] = log_entry.get('learner', 'unknown')
                    trial_info['validation_loss'] = log_entry.get('validation_loss', float('nan'))
                    trial_info['wall_clock_time'] = log_entry.get('wall_clock_time', float('nan'))
                    
                    # config内の情報も一部抽出 (例として数個)
                    config = log_entry.get('config', {})
                    trial_info['config_n_estimators'] = config.get('n_estimators')
                    trial_info['config_num_leaves'] = config.get('num_leaves')
                    trial_info['config_learning_rate'] = config.get('learning_rate')
                    # 他にも必要なconfigパラメータがあれば同様に追加できます
                    
                    if trial_info['learner'] != 'unknown' and not pd.isna(trial_info['validation_loss']):
                        rows.append(trial_info)
                        
                except json.JSONDecodeError:
                    # print(f"[DEBUG] 行 {line_number} は有効なJSONではありませんでした。スキップします: {line_content[:100]}...")
                    pass # JSONとしてパースできない行はスキップ
                except Exception as e:
                    # print(f"[DEBUG] 行 {line_number} の処理中に予期せぬエラー: {e}")
                    pass

    except FileNotFoundError:
        print(f"[ERR] ログファイルが見つかりません: {log_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERR] ログファイルの読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()

    if not rows:
        print("[WARN] ログファイルから有効な学習履歴を抽出できませんでした。")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    print(f"[INFO] {len(df)} 件の試行履歴を抽出しました。")
    return df

# --- メインの処理 ---
if __name__ == "__main__":
    print(f"FLAMLログファイル ('{LOG_FILE_PATH.name}') からLeaderboard情報を再構築します。")
    
    leaderboard_df = parse_flaml_log_json(LOG_FILE_PATH)
    
    if not leaderboard_df.empty:
        # validation_lossでソート
        leaderboard_df_sorted = leaderboard_df.sort_values("validation_loss")
        
        try:
            leaderboard_df_sorted.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"✓ Leaderboard (全試行履歴) を '{OUTPUT_CSV_PATH.name}' に保存しました。")
            
            # 各学習器の最良スコアも表示してみる (オプション)
            if 'learner' in leaderboard_df_sorted.columns:
                leaderboard_best_per_learner = leaderboard_df_sorted.drop_duplicates("learner", keep="first")
                print("\n--- Leaderboard (各学習器のベストスコアのトップ5) ---")
                print(leaderboard_best_per_learner.head().to_string())
            else:
                print("\n--- Leaderboard (最初の5件) ---")
                print(leaderboard_df_sorted.head().to_string())

        except Exception as e:
            print(f"[ERR] LeaderboardのCSV保存中にエラー: {e}")
            print("\n--- 抽出できた全データ (ソート済み) ---")
            print(leaderboard_df_sorted.to_string()) # エラー時は生のデータを表示
    else:
        print("Leaderboardを生成できませんでした。")