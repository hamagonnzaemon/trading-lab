# create_validation_labels.py
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt # datetimeモジュールを 'dt' という別名でインポート

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DATA_FILE = BASE_DIR / "realtime_bars_20210309_20250530.csv" # MT5から出力した全期間データ
OUTPUT_LABELED_VALIDATION_FILE = BASE_DIR / "validation_data_2024_labeled.csv" # ラベル付けした検証データを保存するファイル名

VALIDATION_START_DATE = "2024-01-01"
VALIDATION_END_DATE = "2024-12-31"

TP_USD_GAIN = 3.0  # 利確幅 (価格がこれだけ上昇したら)
SL_USD_LOSS = 2.0  # 損切り幅 (価格がこれだけ下落したら)
MAX_HOLDING_BARS = 12 # 最大保有期間 (この本数先まで見てTP/SLにかからなければ手仕舞い)
# --- ここまで ---

def apply_labeling_rules(df_period: pd.DataFrame) -> pd.Series:
    """
    指定されたDataFrameに利確・損切りルールを適用して正解ラベルを生成します。
    Args:
        df_period (pd.DataFrame): 'bar_time', 'open', 'high', 'low', 'close' を含むDataFrame
    Returns:
        pd.Series: 各行に対応する正解ラベル (1:成功, 0:失敗, np.nan:判定不能)
    """
    print(f"[INFO] ラベル付けを開始します。対象期間のバー数: {len(df_period)}")
    labels = []
    
    # エントリー価格は、シグナルが出たバーの終値と仮定
    entry_prices_series = df_period['close']

    for i in range(len(df_period)):
        # 最後の MAX_HOLDING_BARS 本については、未来のデータが不足するためラベル付けできない
        if i + MAX_HOLDING_BARS >= len(df_period):
            labels.append(np.nan)
            continue

        entry_price = entry_prices_series.iloc[i]
        tp_target_price = entry_price + TP_USD_GAIN
        sl_target_price = entry_price - SL_USD_LOSS
        
        current_label = np.nan # デフォルトは判定不能
        outcome_found_within_holding_period = False

        # 最大保有期間（次のバーからMAX_HOLDING_BARS本先まで）をチェック
        for k in range(1, MAX_HOLDING_BARS + 1):
            future_bar_index = i + k
            
            future_high = df_period['high'].iloc[future_bar_index]
            future_low = df_period['low'].iloc[future_bar_index]

            # 損切りチェックを優先 (保守的なアプローチ)
            if future_low <= sl_target_price:
                current_label = 0 # 損切りヒットで失敗
                outcome_found_within_holding_period = True
                break 
            
            # 利確チェック
            if future_high >= tp_target_price:
                current_label = 1 # 利確ヒットで成功
                outcome_found_within_holding_period = True
                break
        
        # 最大保有期間内に利確も損切りもヒットしなかった場合
        if not outcome_found_within_holding_period:
            exit_price_at_timeout = df_period['close'].iloc[i + MAX_HOLDING_BARS]
            if exit_price_at_timeout > entry_price:
                current_label = 1 # タイムアウト時に利益が出ていれば成功
            else:
                current_label = 0 # タイムアウト時に利益が出ていなければ失敗
        
        labels.append(current_label)

        if (i + 1) % 1000 == 0: # 1000行ごとに進捗を表示
            print(f"[INFO] {i + 1} / {len(df_period)} 件のラベル付け処理完了...")
            
    print("[INFO] ラベル付け処理が完了しました。")
    return pd.Series(labels, index=df_period.index)


if __name__ == "__main__":
    print(f"[INFO] 履歴データファイル '{HISTORICAL_DATA_FILE.name}' を読み込んでいます...")
    try:
        df_full_history = pd.read_csv(HISTORICAL_DATA_FILE)
    except FileNotFoundError:
        print(f"[ERR] 履歴データファイルが見つかりません: {HISTORICAL_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"[ERR] 履歴データファイルの読み込み中にエラー: {e}")
        exit()

    # bar_timeをdatetime型に変換し、タイムゾーン処理
    print("[INFO] 'bar_time' カラムを処理しています...")
    try:
        df_full_history['bar_time'] = pd.to_datetime(df_full_history['bar_time'])
        if df_full_history['bar_time'].dt.tz is None:
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_localize('UTC')
        else:
            df_full_history['bar_time'] = df_full_history['bar_time'].dt.tz_convert('UTC')
    except Exception as e:
        print(f"[ERR] 'bar_time' カラムの処理中にエラー: {e}")
        exit()

    # 検証用期間でデータをフィルタリング
    print(f"[INFO] 検証用期間 ({VALIDATION_START_DATE} ～ {VALIDATION_END_DATE}) のデータを抽出しています...")
    validation_period_mask = (df_full_history['bar_time'] >= pd.to_datetime(VALIDATION_START_DATE, utc=True)) & \
                             (df_full_history['bar_time'] <= pd.to_datetime(VALIDATION_END_DATE, utc=True).replace(hour=23, minute=59, second=59))
    
    df_validation_period = df_full_history[validation_period_mask].copy()

    if df_validation_period.empty:
        print(f"[ERR] 指定された検証期間にデータが見つかりませんでした。日付を確認してください。")
    else:
        print(f"[INFO] 検証用データとして {len(df_validation_period)} 行を抽出しました。")
        
        # 正解ラベルを付与
        df_validation_period['actual_outcome'] = apply_labeling_rules(df_validation_period)
        
        # ラベルが付与できなかった行(NaN)を削除
        original_rows = len(df_validation_period)
        df_validation_period.dropna(subset=['actual_outcome'], inplace=True)
        removed_rows = original_rows - len(df_validation_period)
        if removed_rows > 0:
            print(f"[INFO] ラベルが付与できなかった {removed_rows} 行を削除しました（期間の終端付近）。")

        if df_validation_period.empty:
            print(f"[ERR] ラベル付け後、有効なデータが残りませんでした。")
        else:
            # 結果をCSVに保存
            try:
                df_validation_period.to_csv(OUTPUT_LABELED_VALIDATION_FILE, index=False, encoding='utf-8-sig')
                print(f"✓ ラベル付けされた検証用データを '{OUTPUT_LABELED_VALIDATION_FILE.name}' に保存しました。")
                print(f"  保存されたデータ行数: {len(df_validation_period)}")
                print(f"  成功(1)ラベルの数: {int(df_validation_period['actual_outcome'].sum())}")
                print(f"  失敗(0)ラベルの数: {len(df_validation_period) - int(df_validation_period['actual_outcome'].sum())}")

            except Exception as e:
                print(f"[ERR] ラベル付け済み検証用データのCSV保存中にエラー: {e}")