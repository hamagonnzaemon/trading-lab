#!/usr/bin/env python3
# backtest_oos.py - アウトオブサンプルデータ(2025年5月)でバックテストを行うスクリプト

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt # datetimeモジュールを 'dt' という別名でインポート
from typing import Optional, List # List を明示的にインポート

# --- 設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_flaml_model_v2.pkl"
THRESHOLD_FILE = BASE_DIR / "best_threshold_v2.txt"
HISTORICAL_DATA_FILE = BASE_DIR / "realtime_bars_20210309_20250530.csv" # 新しい全期間データファイル
TRADES_OUTPUT_FILE = BASE_DIR / "oos_backtest_trades.csv"

TEST_PERIOD_START = "2025-05-01"
TEST_PERIOD_END = "2025-05-31" # 5月末まで

# 取引パラメータ (backtest_flaml.py から引用、必要に応じて調整)
LOT    = 1.0
SPREAD_PIPS = 2.0 # スプレッド(pips) XMなら1.5など。価格に直接影響する場合はpipsではなく価格差で。
                  # この簡易バックテストでは価格に直接加減算する方式を取ります。
                  # SymbolInfoDouble(_Symbol, SYMBOL_POINT) を使ったpips換算が必要な場合は別途関数化推奨

# --- モデルの列名取得関数 (realtime_signal_generator.py から引用・調整) ---
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

# --- 特徴量生成関数 (realtime_signal_generator.py の簡易特徴量生成部分をベースに調整) ---
def generate_features_for_backtest(df_period_raw: pd.DataFrame, model_expected_cols: list[str]) -> Optional[pd.DataFrame]:
    if df_period_raw.empty:
        print("[WARN] generate_features_for_backtest に渡されたDataFrameが空です。")
        return None
    
    df_processed = df_period_raw.copy()

    # bar_time を datetime型に (まだなら)
    if not pd.api.types.is_datetime64_any_dtype(df_processed["bar_time"]):
        try:
            df_processed["bar_time"] = pd.to_datetime(df_processed["bar_time"])
        except Exception as e:
            print(f"[ERR] bar_time をdatetimeに変換できませんでした: {e}")
            return None
    
    # --- CSVにない、簡単な特徴量のみをPython側で計算 ---
    # (テクニカル指標はCSVに既にある前提)
    df_processed["body"] = (df_processed["close"] - df_processed["open"]).abs()
    df_processed["upper_wick"]  = df_processed["high"] - np.maximum(df_processed["open"], df_processed["close"])
    df_processed["lower_wick"]  = np.minimum(df_processed["open"], df_processed["close"]) - df_processed["low"]
    df_processed["entry_price"] = df_processed["open"] # モデルが学習した特徴量名に合わせる
    df_processed["exit_price"] = df_processed["close"]  # モデルが学習した特徴量名に合わせる
    
    # --- 時間に関する特徴量 (one-hotエンコーディング) ---
    df_processed["hour"] = df_processed["bar_time"].dt.hour
    df_processed["weekday"] = df_processed["bar_time"].dt.weekday

    for h_val in range(24):
        col_name = f"hour_{h_val}"
        if col_name in model_expected_cols: # モデルが期待する列のみ生成
            df_processed[col_name] = (df_processed["hour"] == h_val).astype(int)
    
    # weekdayは0-4 (月-金) を想定。モデルが weekday_0 から weekday_4 を期待する場合
    for wd_val in range(5): 
        col_name = f"weekday_{wd_val}"
        if col_name in model_expected_cols:
            df_processed[col_name] = (df_processed["weekday"] == wd_val).astype(int)
            
    # --- モデルが期待するカラムが存在するか確認し、不足分はNaNや0で埋める (より安全に) ---
    for col in model_expected_cols:
        if col not in df_processed.columns:
            print(f"[WARN] 期待される特徴量 '{col}' が生成されていません。0.0で埋めます。")
            df_processed[col] = 0.0 # または np.nan

    try:
        # モデルが学習した通りのカラム順で、必要な特徴量を返す
        return df_processed[model_expected_cols]
    except KeyError as e:
        missing_cols = [col for col in model_expected_cols if col not in df_processed.columns]
        print(f"[ERR] 特徴量の最終選択中にカラムが不足しています: {missing_cols}。エラー: {e}")
        return None
    except Exception as e:
        print(f"[ERR] 特徴量生成の最終処理中に予期せぬエラー: {e}")
        return None

# --- バックテスト実行関数 ---
def run_backtest():
    print("[INFO] バックテストを開始します...")

    # モデルと閾値のロード
    try:
        model = joblib.load(MODEL_FILE)
        threshold = float(Path(THRESHOLD_FILE).read_text().strip())
        print(f"[INFO] モデル ({MODEL_FILE.name}) と閾値 ({threshold}) をロードしました。")
    except Exception as e:
        print(f"[FATAL] モデルまたは閾値のロードに失敗: {e}")
        return

    # モデルが期待する特徴量リストを取得
    model_features = get_model_feature_names(MODEL_FILE)
    if not model_features:
        print("[FATAL] モデルから特徴量リストを取得できませんでした。")
        return
    print(f"[INFO] モデルが要求する特徴量 ({len(model_features)}個) を取得しました。")

    # 全期間の履歴データをロード
    try:
        df_historical_full = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['bar_time'])
        print(f"[INFO] 全履歴データ ({HISTORICAL_DATA_FILE.name}) をロードしました。{len(df_historical_full)}行。")

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ bar_time カラムを UTC の aware datetime に変換 ★★★
        if df_historical_full['bar_time'].dt.tz is None:
            # ナイーブな場合はUTCとしてローカライズ
            df_historical_full['bar_time'] = df_historical_full['bar_time'].dt.tz_localize('UTC')
            print("[INFO] 'bar_time' カラムをナイーブからUTC(aware)に変換しました。")
        else:
            # 既にタイムゾーン情報がある場合はUTCに統一
            df_historical_full['bar_time'] = df_historical_full['bar_time'].dt.tz_convert('UTC')
            print("[INFO] 'bar_time' カラムをUTC(aware)に正規化しました。")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    except Exception as e:
        print(f"[FATAL] 履歴データのロードまたは日時変換に失敗: {e}")
        return

    # テスト期間でフィルタリング
    df_test_period_raw = df_historical_full[
        (df_historical_full['bar_time'] >= pd.to_datetime(TEST_PERIOD_START, utc=True)) &
        (df_historical_full['bar_time'] <= pd.to_datetime(TEST_PERIOD_END, utc=True).replace(hour=23, minute=59, second=59)) # 最終日まで含める
    ].copy()

    if df_test_period_raw.empty:
        print(f"[WARN] 指定されたテスト期間 ({TEST_PERIOD_START} ～ {TEST_PERIOD_END}) にデータがありません。")
        return
    print(f"[INFO] テスト期間のデータ ({len(df_test_period_raw)}行) を抽出しました。")
    
    # テスト期間の価格データ（バックテスト計算用）と特徴量データを作成
    # df_test_period_raw にはOHLCVとEAが計算したテクニカル指標が既に入っている前提
    price_data_test = df_test_period_raw.set_index('bar_time') # 時刻をインデックスにすると扱いやすい
    
    print("[INFO] テストデータの特徴量を生成中...")
    X_test = generate_features_for_backtest(df_test_period_raw.copy(), model_features) # df_test_period_raw全体を渡す

    if X_test is None or X_test.empty:
        print("[ERR] テストデータの特徴量生成に失敗しました。バックテストを中止します。")
        return
    
    # モデルの予測時と互換性を持たせるために、X_testのインデックスをprice_data_testに合わせる
    # generate_features_for_backtest が bar_time を含む場合、それをインデックスにできる
    # もし generate_features_for_backtest が bar_time を返さない場合は、元の df_test_period_raw のインデックスを使う必要がある
    if 'bar_time' in X_test.columns: # generate_features が bar_time を含んで返した場合
        X_test = X_test.set_index('bar_time')
    else: # generate_features が bar_time を返さない場合 (元のインデックスを維持)
        X_test.index = price_data_test.index[-len(X_test):] # generate_featuresでdropnaされた場合を考慮して長さを合わせる

    # カラム名を文字列に統一
    X_test.columns = X_test.columns.astype(str)
    print(f"[INFO] 特徴量生成完了。予測に使用するデータ: {len(X_test)}行")


    # 予測確率の取得
    try:
        print("[INFO] モデルによる予測確率を計算中...")
        proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"[ERR] 予測確率の計算中にエラー: {e}")
        print("[DEBUG] X_test head:", X_test.head())
        print("[DEBUG] X_test columns:", X_test.columns.tolist())
        print("[DEBUG] Model expected features:", model_features)
        return

    # X_test (インデックスがbar_time) に確率とシグナルを追加
    df_signals_test = X_test.copy()
    df_signals_test['probability'] = proba
    df_signals_test['signal'] = (df_signals_test['probability'] >= threshold).astype(int)
    
    # --- バックテスト実行ロジック (以前の backtest_flaml.py のロジックをベースに) ---
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    
    # スプレッドを価格単位に変換 (ポイント単位ではない、実際の価格差として)
    # これは通貨ペアに依存するので、正確な値は別途確認・設定が必要
    # 例: GOLDなら1pips = 0.1, EURUSDなら1pips=0.0001 など
    # ここでは SPREAD_PIPS が価格差そのものだと仮定するか、より正確な変換を実装する
    # 簡単のため、ここでは SPREAD_PIPS が価格に直接加減算できる値だとしておく
    # (より正確には point_value * SPREAD_PIPS のような計算が必要)
    spread_cost = SPREAD_PIPS # 仮。実際には SymbolInfoDouble(_Symbol, SYMBOL_POINT) * SPREAD_PIPS

    print("[INFO] バックテストのシミュレーションを開始します...")
    # price_data_test と df_signals_test をマージして、シグナルと価格を同時に扱えるようにする
    # インデックス (bar_time) が一致している必要がある
    df_for_backtest = price_data_test.join(df_signals_test[['signal']], how='inner')
    
    if df_for_backtest.empty:
        print("[ERR] 価格データとシグナルデータのマージに失敗、または結果が空です。")
        return

    for i in range(len(df_for_backtest) - 1): # 最後から2番目のバーまでループ（最後のバーで決済するため）
        current_row = df_for_backtest.iloc[i]
        next_row = df_for_backtest.iloc[i+1] # 決済用の次のバー

        if not in_position and current_row["signal"] == 1: # BUYシグナル
            entry_price = current_row["close"] + spread_cost # 現在の終値でエントリー (+スプレッドコスト)
            entry_time  = current_row.name # インデックス (bar_time)
            in_position = True
            # print(f"[TRADE] Entry: {entry_time} at {entry_price}")

        elif in_position: # ポジションを持っている場合、次のバーの始値で決済
            exit_price = next_row["open"] - spread_cost # 次のバーの始値で決済 (-スプレッドコスト)
            exit_time  = next_row.name   # インデックス (bar_time)
            
            pnl = (exit_price - entry_price) * LOT # 簡単な損益計算 (ロングのみ)
            trades.append({
                "entry_time": entry_time, "exit_time": exit_time,
                "entry_price": entry_price, "exit_price": exit_price,
                "pnl": pnl
            })
            in_position = False
            # print(f"[TRADE] Exit: {exit_time} at {exit_price}, PnL: {pnl:.2f}")

    if not trades:
        print("[INFO] バックテスト期間中に取引は発生しませんでした。")
        return

    bt_results = pd.DataFrame(trades)
    bt_results["cum_pnl"] = bt_results["pnl"].cumsum()

    # --- 結果表示 ---
    print("\n--- バックテスト結果 (アウトオブサンプル: 2025年5月) ---")
    total_pnl = bt_results["pnl"].sum()
    num_trades = len(bt_results)
    wins = (bt_results["pnl"] > 0).sum()
    losses = (bt_results["pnl"] <= 0).sum() # <= 0 を負けとカウント
    win_rate = wins / num_trades if num_trades > 0 else 0
    
    max_drawdown = (bt_results["cum_pnl"].cummax() - bt_results["cum_pnl"]).max()
    if pd.isna(max_drawdown): max_drawdown = 0 # 取引がない場合などを考慮

    print(f"取引回数         : {num_trades}")
    print(f"総損益           : {total_pnl:.2f}")
    print(f"勝率             : {win_rate:.2%}")
    print(f"勝ちトレード数   : {wins}")
    print(f"負けトレード数   : {losses}")
    print(f"平均利益 (勝ち)  : {bt_results[bt_results['pnl'] > 0]['pnl'].mean():.2f}" if wins > 0 else "N/A")
    print(f"平均損失 (負け)  : {bt_results[bt_results['pnl'] <= 0]['pnl'].mean():.2f}" if losses > 0 else "N/A")
    print(f"最大ドローダウン : {max_drawdown:.2f}")

    try:
        bt_results.to_csv(TRADES_OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✓ 取引履歴を '{TRADES_OUTPUT_FILE.name}' に保存しました。")
    except Exception as e:
        print(f"[ERR] 取引履歴のCSV保存に失敗: {e}")

if __name__ == "__main__":
    run_backtest()