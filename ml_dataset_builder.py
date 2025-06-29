#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import numpy as np

#──────────────────────────────────────────────────────────────
#  ① フルパスでファイルを指定（VSCode からも引数不要）
#──────────────────────────────────────────────────────────────
# ユーザーの環境に合わせてパスを修正してください
BASE_DIR    = Path("/Users/hamadakou/Desktop/trade_log") # ユーザーの環境の基準パス
BARS_CSV    = BASE_DIR / "GOLD_M5_202101040100_202504302355.csv"
TRADES_CSV  = BASE_DIR / "generated_entry_log.csv" # このファイル名もユーザーの環境に合わせる
OUTPUT_CSV  = BASE_DIR / "trades_ml_roadmap.csv"

#──────────────────────────────────────────────────────────────
#  ② 足データ読み込み：タブ区切りで「日付」「時刻」を別列取得
#──────────────────────────────────────────────────────────────
print(f"Loading BARS_CSV: {BARS_CSV} ...")
try:
    tmp = pd.read_csv(
        BARS_CSV,
        sep=r"\t",              # ← タブ区切り
        engine="python",
        names=[
            "date", "time",
            "open","high","low","close",
            "tickvol","vol","spread"
        ],
        header=0                # ヘッダー行を飛ばす
    )
    # 日付＋時刻をくっつけて正しく datetime 型に
    tmp["datetime"] = pd.to_datetime(
        tmp["date"] + " " + tmp["time"],
        format="%Y.%m.%d %H:%M:%S",
        errors='coerce' # パースエラー時は NaT にする
    )
    # NaTになった行があれば警告し、削除
    if tmp["datetime"].isna().any():
        print(f"警告: {BARS_CSV} の日付/時刻パースで NaT が発生しました。該当行を削除します。")
        print(tmp[tmp["datetime"].isna()]) # NaTになった行を表示
        tmp.dropna(subset=["datetime"], inplace=True)

    bars = tmp.set_index("datetime").drop(columns=["date","time"])

    # ===== Debug Point 1: GOLD_M5_...csv パース後の bars.index (修正版) =====
    print("--- Debug Point 1: GOLD_M5_...csv パース後の bars.index ---")
    if not bars.empty and isinstance(bars.index, pd.DatetimeIndex):
        print("bars.index (先頭3件):\n", bars.index[:3]) # スライスで表示
        print("bars.index (末尾3件):\n", bars.index[-3:]) # スライスで表示
        print(f"bars.index.dtype: {bars.index.dtype}")
        print(f"bars.index に NaT (Not a Time) が含まれる数: {bars.index.isna().sum()}")
    elif bars.empty:
        print("bars データフレームが空です（ファイル読み込み失敗またはデータなし）。")
    else:
        print(f"bars.index が期待される DatetimeIndex ではありません。現在の型: {type(bars.index)}")
    print("--- Debug Point 1 End ---")
    # ===== Debug Point 1 End =====

except FileNotFoundError:
    print(f"エラー: BARS_CSV ({BARS_CSV}) が見つかりません。")
    exit()
except Exception as e:
    print(f"エラー: BARS_CSV ({BARS_CSV}) の読み込みまたは処理中に問題が発生しました: {e}")
    exit()


#──────────────────────────────────────────────────────────────
#  ③ テクニカル指標の計算関数
#──────────────────────────────────────────────────────────────
def calc_stochastic(df, k_period=15, slowing_period=3, d_period=9):
    """ストキャスティクス (%K, %D) を計算 (STO_CLOSECLOSE & MODE_SMA 相当)"""
    low_k = df['close'].rolling(window=k_period).min()
    high_k = df['close'].rolling(window=k_period).max()
    
    fast_k_numerator = df['close'] - low_k
    fast_k_denominator = high_k - low_k
    fast_k = (fast_k_numerator / fast_k_denominator) * 100
    
    fast_k.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    slow_k = fast_k.rolling(window=slowing_period).mean()
    slow_d = slow_k.rolling(window=d_period).mean()
    
    return slow_k, slow_d

#──────────────────────────────────────────────────────────────
#  ④ トレードシミュレーションと特徴量エンジニアリング
#──────────────────────────────────────────────────────────────
def simulate_trades(trades_log_csv, df, out_csv,
                    stop_loss_pips=150, take_profit_pips=150):
    print(f"Loading TRADES_CSV: {trades_log_csv} ...")
    try:
        trades_log = pd.read_csv(trades_log_csv, parse_dates=["entry_time"])
    except FileNotFoundError:
        print(f"エラー: TRADES_CSV ({trades_log_csv}) が見つかりません。")
        return pd.DataFrame()
    except Exception as e:
        print(f"エラー: TRADES_CSV ({trades_log_csv}) の読み込み中に問題が発生しました: {e}")
        return pd.DataFrame()

    trades_out = []
    for _, trade_signal in trades_log.iterrows():
        entry_bar_time = trade_signal["entry_time"]
        if entry_bar_time not in df.index:
            continue
        entry_row = df.loc[entry_bar_time]
        
        entry_time  = entry_bar_time
        bar_time    = entry_bar_time
        entry_price = entry_row["open"]

        entry_feats = {
            "ema_fast": entry_row.get("ema_fast", np.nan),
            "ema_mid":  entry_row.get("ema_mid", np.nan),
            "ema_slow": entry_row.get("ema_slow", np.nan),
            "K":        entry_row.get("K", np.nan),
            "D":        entry_row.get("D", np.nan)
        }
        
        pip_value = 0.01
        sl_diff = stop_loss_pips * pip_value
        tp_diff = take_profit_pips * pip_value
        trade_type = trade_signal.get("trade_type", 1)

        if trade_type == 1:
            sl_level = entry_price - sl_diff
            tp_level = entry_price + tp_diff
        else:
            sl_level = entry_price + sl_diff
            tp_level = entry_price - tp_diff
        
        exit_time   = pd.NaT
        exit_price  = np.nan
        label       = np.nan
        
        try:
            start_index = df.index.get_loc(entry_bar_time)
        except KeyError:
            continue

        max_holding_bars = 200
        for i in range(1, max_holding_bars + 1):
            current_bar_index_pos = start_index + i
            if current_bar_index_pos >= len(df.index):
                break
            current_bar_time = df.index[current_bar_index_pos]
            current_row = df.iloc[current_bar_index_pos]
            hh, ll = current_row["high"], current_row["low"]

            if trade_type == 1:
                if ll <= sl_level:
                    exit_time, exit_price, label = current_bar_time, sl_level, 0
                    break
                elif hh >= tp_level:
                    exit_time, exit_price, label = current_bar_time, tp_level, 1
                    break
            else:
                if hh >= sl_level:
                    exit_time, exit_price, label = current_bar_time, sl_level, 0
                    break
                elif ll <= tp_level:
                    exit_time, exit_price, label = current_bar_time, tp_level, 1
                    break
        
        trades_out.append({
            "entry_time":  entry_time,
            "bar_time":    bar_time,
            "exit_time":   exit_time,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "label":       label,
            **entry_feats
        })

    df_final_trades = pd.DataFrame(trades_out)

    # ===== Debug Point 2: trades_ml_roadmap.csv 書き出し直前の df_final_trades['bar_time'] =====
    print("--- Debug Point 2: trades_ml_roadmap.csv 書き出し直前の df_final_trades['bar_time'] ---")
    if not df_final_trades.empty and 'bar_time' in df_final_trades.columns:
        print("df_final_trades['bar_time'].head(3):\n", df_final_trades['bar_time'].head(3))
        print("df_final_trades['bar_time'].tail(3):\n", df_final_trades['bar_time'].tail(3))
        print(f"df_final_trades['bar_time'].dtype: {df_final_trades['bar_time'].dtype}")
        print(f"df_final_trades['bar_time'] に NaT/NaN が含まれる数: {df_final_trades['bar_time'].isna().sum()}")
    elif df_final_trades.empty:
        print("出力する df_final_trades データフレームが空です。")
    else:
        print("出力する df_final_trades データフレームに 'bar_time' カラムが存在しません。")
    print("--- Debug Point 2 End ---")
    # ===== Debug Point 2 End =====

    df_final_trades.to_csv(out_csv, index=False)
    print(f"✅ {len(df_final_trades)} 件のトレードを “{out_csv}” に出力しました。")
    return df_final_trades

#──────────────────────────────────────────────────────────────
#  ⑤ メイン処理
#──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if 'bars' not in locals() or bars.empty:
        print("エラー: 'bars' データフレームが正しく初期化されていません。処理を中止します。")
        exit()
        
    print("メイン処理開始: テクニカル指標を計算中...")
    bars['ema_fast'] = bars['close'].ewm(span=20, adjust=False).mean()
    bars['ema_mid']  = bars['close'].ewm(span=50, adjust=False).mean()
    bars['ema_slow'] = bars['close'].ewm(span=70, adjust=False).mean()
    bars['K'], bars['D'] = calc_stochastic(bars)
    print("テクニカル指標の計算完了。")

    simulated_trades_df = simulate_trades(TRADES_CSV, bars, OUTPUT_CSV)

    if simulated_trades_df.empty:
        print("警告: トレードシミュレーションの結果、データが出力されませんでした。")
    else:
        print(f"メイン処理完了: {OUTPUT_CSV} が生成されました（または上書きされました）。")