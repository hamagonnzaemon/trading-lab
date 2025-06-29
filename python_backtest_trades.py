#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import pandas as pd
import numpy as np
from pathlib import Path



# ──────────────────────────────────────────────────────────────
#  ◆ デフォルト設定
# ──────────────────────────────────────────────────────────────
DEFAULT_BARS     = Path.home() / "Desktop/trade_log/GOLD_M5_202101040100_202504302355.csv"
DEFAULT_OUT      = Path.home() / "Desktop/trade_log/generated_entry_log.csv"
DEFAULT_PIP_SIZE = 0.1



# ──────────────────────────────────────────────────────────────
#  ◆ バックテスト＆エクスポート関数
# ──────────────────────────────────────────────────────────────
def backtest_and_export(
    bars_csv: Path,
    out_csv:  Path,
    *,
    ema_fast: int    = 20,
    ema_mid:  int    = 50,
    ema_slow: int    = 70,
    stoch_k:   int   = 15,
    stoch_d:   int   = 3,
    stoch_slow:int   = 9,
    cross_thr: float = 2.0,
    sl_pips:   float = 20.0,
    tp_pips:   float = 30.0,
    pip_size:  float = DEFAULT_PIP_SIZE
):
    """
    最新の EA ロジックを Python で再現し、
    entry_time, bar_time, exit_time, entry_price, exit_price, label
    に加えてテクニカル指標を含む CSV を出力。
    """

       # ──【追加箇所①】──────────────────────────────────────────────
   # Debug: ファイル存在＆先頭数行の確認
    if not bars_csv.exists():
        raise FileNotFoundError(f"✖ ファイルが存在しません: {bars_csv}")
    else:
        print(f"✅ 読み込むファイル: {bars_csv}")
      # ──【追加デバッグ】── 先頭5行だけ 本番同様に skiprows=1・空白区切りで読む
    tmp = pd.read_csv(
       bars_csv,
       delim_whitespace=True, engine='python',
       names=["datetime","open","high","low","close","tickvol","vol","spread"],
       header=None, skiprows=1, nrows=5
   )
    print("── bars.head() ─────────────────────────────")
    print(tmp)
    print("columns:", tmp.columns.tolist())
    # インデックス設定前の型を確認
    print("datetime dtype (before parse):", tmp["datetime"].dtype)
    # ──【追加箇所① ここまで】──────────────────────────────────


    # 1) M5 足読み込み
  # ── タブ区切りで日付(date)と時刻(time)を分離読み込み
    raw = pd.read_csv(
        bars_csv,
        sep="\t", engine="python",
        names=["date","time","open","high","low","close","tickvol","vol","spread"],
        header=None, skiprows=1
    )
    # 文字列結合して datetime 型に変換
    raw["datetime"] = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y.%m.%d %H:%M:%S")
    bars = raw.set_index("datetime")[["open","high","low","close","tickvol","vol","spread"]]
    
        # ──【②】読み込み後サンプル確認──
    print("── bars.head() ─────────────────────────")
    print(bars.head())
    print("── bars.dtypes ─────────────────────────")
    print(bars.dtypes)

    # 2) テクニカル指標を計算
    bars["ema_fast"] = bars["close"].ewm(span=ema_fast, adjust=False).mean()
    bars["ema_mid" ] = bars["close"].ewm(span=ema_mid , adjust=False).mean()
    bars["ema_slow"] = bars["close"].ewm(span=ema_slow, adjust=False).mean()

    low_min  = bars["low"].rolling(window=stoch_k).min()
    high_max = bars["high"].rolling(window=stoch_k).max()
    raw_k    = 100 * (bars["close"] - low_min) / (high_max - low_min)
    slow_k   = raw_k.rolling(window=stoch_slow).mean()
    bars["K"] = slow_k
    bars["D"] = slow_k.rolling(window=stoch_d).mean()

    # 3) バックテストループ
    trades = []
    armed       = False
    in_position = False
    prev_K, prev_D = None, None

    sl_dist = sl_pips * pip_size
    tp_dist = tp_pips * pip_size

    for time, row in bars.iterrows():
        K = row["K"]; D = row["D"]

        # armed 制御
        if K >= 80.0:   armed = True
        if K <= 20.0:   armed = False

        # ゴールデンクロス検出
        crossed_up  = (prev_K is not None) and (prev_K < prev_D and K >= D)
        valid_cross = crossed_up and ((K - D) >= cross_thr)
        prev_K, prev_D = K, D

        # エントリー
        if not in_position and armed and valid_cross:
            entry_time    = time
            bar_time      = time
            entry_price   = row["close"]
            sl_level      = entry_price - sl_dist
            tp_level      = entry_price + tp_dist
            # テクニカル指標を保存
            entry_ema_fast = row["ema_fast"]
            entry_ema_mid  = row["ema_mid"]
            entry_ema_slow = row["ema_slow"]
            entry_K        = K
            entry_D        = D

            in_position = True
            continue

        # 決済判定
        if in_position:
            hh = row["high"]; ll = row["low"]
            if hh >= tp_level:
                exit_time   = time; exit_price = tp_level; label = 1
            elif ll <= sl_level:
                exit_time   = time; exit_price = sl_level; label = 0
            else:
                continue

            trades.append({
                "entry_time":    entry_time,
                "bar_time":      bar_time,
                "exit_time":     exit_time,
                "entry_price":   entry_price,
                "exit_price":    exit_price,
                "label":         label,
                # ── 追加したテクニカル指標 ───────────────────
                "ema_fast":      entry_ema_fast,
                "ema_mid":       entry_ema_mid,
                "ema_slow":      entry_ema_slow,
                "K":             entry_K,
                "D":             entry_D
            })
            in_position = False
            armed       = False

    # 4) データフレーム化して出力
    df = pd.DataFrame(trades)
    df.to_csv(out_csv, index=False)
    print(f"✅ {len(df)} トレードを '{out_csv}' に出力しました。")

# ──────────────────────────────────────────────────────────────
#  ◆ CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python Backtest & Trade Log Export (with technical columns)"
    )
    parser.add_argument("--bars", required=False,
                        default=str(DEFAULT_BARS),
                        help=f"M5 足 CSV（デフォルト: %(default)s）")
    parser.add_argument("--out",  required=False,
                        default=str(DEFAULT_OUT),
                        help=f"出力ファイル名（デフォルト: %(default)s）")
    parser.add_argument("--pip",  required=False, type=float,
                        default=DEFAULT_PIP_SIZE,
                        help=f"1 pip のサイズ（デフォルト: %(default)s）")

    args = parser.parse_args()

    backtest_and_export(
        bars_csv=Path(args.bars).expanduser(),
        out_csv =Path(args.out).expanduser(),
        pip_size=args.pip
    )
