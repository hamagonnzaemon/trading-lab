#!/usr/bin/env python3
# realtime_signal_generator.py (bar-close driven loop – v2025-05-21c) - 通知＆シグナル履歴保存機能付き

import os
import time
import math
import signal
import joblib
import numpy as np # np.nan を使うためにインポート
import pandas as pd
from pathlib import Path
import datetime as dt # datetimeモジュールを 'dt' という別名でインポート
import subprocess
from typing import Optional

# ────────── ユーザ設定 ──────────
BASE_DIR      = Path("/Users/hamadakou/Desktop/trade_log") # ご自身の環境に合わせてください
MT5_FILES_DIR = Path(
    "/Users/hamadakou/Library/Application Support/net.metaquotes.wine.metatrader5/" # ご自身の環境に合わせてください
    "drive_c/Program Files/MetaTrader 5/MQL5/Files"
)

REALTIME_BARS_FILE       = MT5_FILES_DIR / "realtime_bars_for_python.csv"
SIGNAL_OUTPUT_FILE       = MT5_FILES_DIR / "signal_to_ea.txt"
SIGNAL_OUTPUT_FILE_TEMP  = MT5_FILES_DIR / "signal_to_ea.txt.tmp"

MODEL_FILE     = BASE_DIR / "classweight_model.pkl"#best_flaml_model.pkl（旧モデル）
THRESHOLD_FILE = BASE_DIR / "best_threshold.txt"
SIGNAL_HISTORY_FILE = BASE_DIR / "signal_history.csv" # ★★★ シグナル履歴保存用ファイル ★★★

TF_SEC              = 300        # 5-minute bar
BAR_CLOSE_DELAY_SEC = 2          # バー確定後この秒数待ってから読みに行く
CSV_STABLE_SPAN     = 0.10       # サイズ変化チェック間隔
MAX_WAIT_SEC        = 1.0        # 「書き込み完了待ち」の最長時間
LOOP_POLL_SEC       = 10         # 補助ポーリング（1 バー内で複数回チェック）

LEARNED_FEATURE_COLUMNS = [
    "ema_fast","ema_mid","ema_slow","K","D","open","high","low","close",
    "ATR_14","body_length","upper_wick","lower_wick","KD_angle",
    "hour","weekday",
    *[f"hour_{h}" for h in range(24)],
    *[f"weekday_{d}" for d in range(5)]
]

# ────────── Ctrl-C 停止フラグ ──────────
_STOP = False
def _sigint(sig, frame):
    global _STOP
    if not _STOP:
        print("\n[Loop] Ctrl-C 受信。現在のサイクル完了後に停止します…")
    _STOP = True
signal.signal(signal.SIGINT, _sigint)

# ────────── モデルの列名取得 ──────────
def feature_columns_from_model(model_path: Path, fallback: list[str]) -> list[str]:
    try:
        mdl = joblib.load(model_path)
        if hasattr(mdl, "feature_names_in_"):
             return list(mdl.feature_names_in_)
        if hasattr(mdl, "feature_name_"):
            return list(mdl.feature_name_)
        actual_estimator = None
        if hasattr(mdl, 'model') and mdl.model is not None:
            actual_estimator = mdl.model
        elif hasattr(mdl, 'fitted_estimator') and mdl.fitted_estimator is not None:
             actual_estimator = mdl.fitted_estimator
        else:
            actual_estimator = mdl
        if actual_estimator:
            if hasattr(actual_estimator, "feature_names_in_"):
                return list(actual_estimator.feature_names_in_)
            if hasattr(actual_estimator, "feature_name_"):
                return list(actual_estimator.feature_name_)
            if hasattr(actual_estimator, "booster_") and hasattr(actual_estimator.booster_, "feature_name"):
                return list(actual_estimator.booster_.feature_name())
        print("[WARN] モデルオブジェクトから特徴量名リストを取得できませんでした。フォールバックを使用します。")
    except Exception as e:
        print(f"[WARN] モデル列名取得中にエラーが発生しました: {e}。フォールバックを使用します。")
    return fallback

# ────────── 特徴量生成 (修正版) ──────────
def generate_features(df: pd.DataFrame, cols: list[str]) -> Optional[pd.DataFrame]:
    df_copy = df.copy()
    if df_copy.empty:
        print("[WARN] generate_features に渡されたDataFrameが空です。")
        return None
    if not pd.api.types.is_datetime64_any_dtype(df_copy["bar_time"]):
        try:
            df_copy["bar_time"] = pd.to_datetime(df_copy["bar_time"])
        except Exception as e:
            print(f"[ERR] bar_time をdatetimeに変換できませんでした: {e}")
            return None
    if df_copy["bar_time"].isnull().any():
        print("[WARN] bar_time にnullが含まれています。特徴量生成をスキップします。")
        return None
    try:
        latest_bar_time = df_copy["bar_time"].iloc[-1]
    except IndexError:
        print("[WARN] bar_time が空のため、latest_bar_timeを取得できませんでした。")
        return None
    df_copy["body_length"] = (df_copy["close"] - df_copy["open"]).abs()
    df_copy["upper_wick"]  = df_copy["high"] - np.maximum(df_copy["open"], df_copy["close"])
    df_copy["lower_wick"]  = np.minimum(df_copy["open"], df_copy["close"]) - df_copy["low"]
    h  = int(latest_bar_time.hour)
    wd = int(latest_bar_time.weekday())
    df_copy["hour"] = h
    df_copy["weekday"] = wd
    for col_name_in_list in (item for item in cols if item.startswith("hour_")):
        df_copy[col_name_in_list] = 1 if col_name_in_list == f"hour_{h}" else 0
    for col_name_in_list in (item for item in cols if item.startswith("weekday_")):
        df_copy[col_name_in_list] = 1 if col_name_in_list == f"weekday_{wd}" else 0
    try:
        existing_cols = [col for col in cols if col in df_copy.columns]
        missing_cols = [col for col in cols if col not in df_copy.columns]
        if missing_cols:
            print(f"[WARN] 特徴量生成時、一部カラムが不足 (生成されず): {missing_cols}")
        if not existing_cols:
             print(f"[ERR] 特徴量生成後、必要なカラムが一つも存在しません。要求カラム: {cols}")
             return None
        return df_copy[existing_cols]
    except Exception as e:
        print(f"[ERR] 特徴量生成の最終処理中に予期せぬエラー: {e}")
        return None

# ────────── バークローズ UTC 時刻 ──────────
def next_bar_close_utc(now: dt.datetime) -> dt.datetime:
    ts = math.ceil(now.timestamp() / TF_SEC) * TF_SEC + BAR_CLOSE_DELAY_SEC
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)

# ────────── CSV が「安定」したか判定 ──────────
def _file_is_stable(path: Path, span: float = CSV_STABLE_SPAN) -> bool:
    try:
        size1 = path.stat().st_size
        if size1 == 0:
            return False
        time.sleep(span)
        size2 = path.stat().st_size
        return size1 == size2
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[WARN] _file_is_stableチェック中にエラー ({path}): {e}")
        return False

# ────────── CSV を安全に読む ──────────
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_SEC:
        if path.exists() and _file_is_stable(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
                else:
                    time.sleep(CSV_STABLE_SPAN)
                    continue
            except FileNotFoundError:
                pass
            except pd.errors.EmptyDataError:
                time.sleep(CSV_STABLE_SPAN)
                continue
            except Exception as e:
                print(f"[ERR] CSV読み込み試行中にエラー ({path}): {e}")
                time.sleep(CSV_STABLE_SPAN)
                continue
        if _STOP: return None
        time.sleep(CSV_STABLE_SPAN / 2)
    print(f"[WARN] {path} の安定待機がタイムアウトしました。最終読み込みを試みます。")
    try:
        df = pd.read_csv(path)
        if not df.empty: return df
        print(f"[WARN] タイムアウト後の最終読み込みでも {path} は空でした。")
        return None
    except Exception as e:
        print(f"[ERR] タイムアウト後の最終CSV読み込みに失敗 ({path}): {e}")
        return None

# デスクトップ通知用関数
def send_mac_notification(title: str, message: str, sound_name: Optional[str] = "Glass"):
    try:
        command = ['osascript', '-e', f'display notification "{message}" with title "{title}"']
        if sound_name:
            command.extend(['sound name', sound_name])
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        print(f"[INFO] デスクトップ通知送信成功: {title}")
    except FileNotFoundError:
        print("[WARN] 'osascript' コマンドが見つかりません。この通知機能はmacOSでのみ動作します。")
    except subprocess.TimeoutExpired:
        print(f"[WARN] デスクトップ通知の送信がタイムアウトしました。")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] デスクトップ通知の送信に失敗 (osascript error): {e.stderr if e.stderr else e.stdout if e.stdout else '詳細不明'}")
    except Exception as e:
        print(f"[WARN] デスクトップ通知の送信中に予期せぬエラー: {e}")

# ────────── 1 サイクル処理 ──────────
def run_cycle(model, thr: float, cols: list[str], last_seen: Optional[dt.datetime]):
    df_raw_bar = safe_read_csv(REALTIME_BARS_FILE) # ★★★ 生のバーデータを保持 ★★★
    if df_raw_bar is None:
        return last_seen
    if "bar_time" not in df_raw_bar.columns:
        print(f"[ERR] {REALTIME_BARS_FILE} に 'bar_time' カラムが含まれていません。")
        return last_seen

    try:
        df_raw_bar["bar_time"] = pd.to_datetime(df_raw_bar["bar_time"])
        if df_raw_bar["bar_time"].dt.tz is None:
            bar_time_utc = df_raw_bar["bar_time"].iloc[-1].tz_localize('UTC')
        else:
            bar_time_utc = df_raw_bar["bar_time"].iloc[-1].tz_convert('UTC')
    except Exception as e:
        print(f"[ERR] bar_time の処理中にエラー: {e}")
        return last_seen

    if last_seen is not None and bar_time_utc <= last_seen:
        return last_seen

    #print(f"[INFO] 新しいバーを処理中: {bar_time_utc}")
    # 特徴量生成に渡すのはDataFrameの最後の1行のコピー
    current_bar_features_input = df_raw_bar.tail(1).copy() # ★★★ 特徴量生成の入力 ★★★
    X = generate_features(current_bar_features_input, cols)
    
    if X is None or X.empty:
        print("[WARN] 特徴量生成に失敗したか、結果が空でした。このバーの処理をスキップします。")
        return bar_time_utc

    try:
        proba = model.predict_proba(X.head(1))[0, 1]
        cmd   = "BUY" if proba >= thr else "NONE"
    except Exception as e:
        print(f"[ERR] モデル予測中にエラー: {e}")
        return bar_time_utc

    try:
        SIGNAL_OUTPUT_FILE_TEMP.write_text(cmd)
        os.replace(SIGNAL_OUTPUT_FILE_TEMP, SIGNAL_OUTPUT_FILE)
    except Exception as e:
        print(f"[ERR] シグナルファイル '{SIGNAL_OUTPUT_FILE}' 書き込み失敗: {e}")

    print(f"{bar_time_utc}  P={proba:.4f}  -> {cmd}")

    if cmd == "BUY":
        print(f"[ALERT] ★★★ BUYシグナル発生！ ★★★ (P={proba:.4f})")
        send_mac_notification(
            title="リアルタイム売買シグナル",
            message=f"BUYシグナル発生！\n時刻: {bar_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n確率: {proba:.4f}"
        )
    
    # ★★★ シグナル履歴の保存処理 (ここに追加) ★★★
    try:
        # 保存するバーのデータ (OHLCVなど) を取得
        # current_bar_features_input には元のOHLCVが含まれているはず
        latest_ohlcv_data = current_bar_features_input.iloc[-1]

        log_entry = {
            "timestamp_utc": bar_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f%z'), # 文字列として保存推奨
            "open": latest_ohlcv_data.get("open", pd.NA),
            "high": latest_ohlcv_data.get("high", pd.NA),
            "low": latest_ohlcv_data.get("low", pd.NA),
            "close": latest_ohlcv_data.get("close", pd.NA),
            "volume": latest_ohlcv_data.get("volume", latest_ohlcv_data.get("tick_volume", pd.NA)), # 'volume' or 'tick_volume'
            "model_probability": proba,
            "predicted_signal": cmd,
            "actual_outcome": np.nan # 後で結果を追記するためのプレースホルダー
        }
        log_df = pd.DataFrame([log_entry])

        if not SIGNAL_HISTORY_FILE.exists():
            log_df.to_csv(SIGNAL_HISTORY_FILE, index=False, mode='w', header=True, encoding='utf-8-sig')
        else:
            log_df.to_csv(SIGNAL_HISTORY_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')
        # print(f"[INFO] シグナル履歴を {SIGNAL_HISTORY_FILE.name} に追記しました。") # 毎回のログは冗長かもしれないのでコメントアウト
    except Exception as e:
        print(f"[ERR] シグナル履歴のCSV ({SIGNAL_HISTORY_FILE.name}) 書き込みに失敗: {e}")
    
    return bar_time_utc

# ────────── メインループ ──────────
def main():
    print("[INFO] プログラムを開始します。")
    try:
        model_path = MODEL_FILE
        threshold_path = THRESHOLD_FILE
        if not model_path.exists():
            print(f"[FATAL] モデルファイルが見つかりません: {model_path}")
            return
        if not threshold_path.exists():
            print(f"[FATAL] 閾値ファイルが見つかりません: {threshold_path}")
            return
            
        model     = joblib.load(model_path)
        threshold = float(threshold_path.read_text().strip())
        print(f"[INFO] モデル ({model_path.name}) と閾値 ({threshold}) をロードしました。")
    except Exception as e:
        print(f"[FATAL] モデルまたは閾値のロードに失敗: {e}")
        return

    cols = feature_columns_from_model(MODEL_FILE, LEARNED_FEATURE_COLUMNS)
    if not cols:
        print("[FATAL] 使用する特徴量のカラム名リストを取得できませんでした。")
        return
    print(f"[INFO] 使用する特徴量カラム ({len(cols)}個): {cols[:3]}...{cols[-2:]} など")

    print(f"[Loop] Realtime signal generator (TF={TF_SEC//60}min, Delay={BAR_CLOSE_DELAY_SEC}s) started. Ctrl-C で停止。")

    last_seen: Optional[dt.datetime] = None
    
    while not _STOP:
        try:
            current_utc_time = dt.datetime.now(dt.timezone.utc)
            target_bar_close_time = next_bar_close_utc(current_utc_time)
            sleep_duration = (target_bar_close_time - current_utc_time).total_seconds()
            
            if sleep_duration > 0:
                wait_until = time.monotonic() + sleep_duration
                while time.monotonic() < wait_until and not _STOP:
                    time_to_wait_this_loop = min(LOOP_POLL_SEC, wait_until - time.monotonic())
                    if time_to_wait_this_loop > 0.01:
                        time.sleep(time_to_wait_this_loop)
                    else:
                        break
            
            if _STOP: break

            last_seen = run_cycle(model, threshold, cols, last_seen)

        except Exception as e:
            print(f"[ERR] メインループで予期せぬエラーが発生しました: {e}")
            print("[INFO] 10秒後に処理を再試行します...")
            for _ in range(10):
                if _STOP: break
                time.sleep(1)
            if _STOP: break
            
    print("[Loop] 停止処理を完了しました。プログラムを終了します。")

if __name__ == "__main__":
    main()