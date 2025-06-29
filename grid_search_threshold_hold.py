#!/usr/bin/env python3
#「閾値 & 保持バー」を一括テストする簡易コード
import pandas as pd, joblib, numpy as np, itertools, argparse
from pathlib import Path
from simulate_threshold_backtest import enrich_features, LEARNED_COLS

def run_grid(bars_csv, model_pkl, thr_list, hold_list):
    df   = enrich_features(pd.read_csv(bars_csv, parse_dates=['bar_time']))
    mdl  = joblib.load(model_pkl)
    prob = mdl.predict_proba(df[LEARNED_COLS])[:,1]
    df['proba'] = prob

    results = []
    for thr, hold in itertools.product(thr_list, hold_list):
        sig = (prob >= thr).astype(int)
        pnl = []
        for i, flag in enumerate(sig):
            if flag:
                if i+hold >= len(df): break
                pnl.append(df.loc[i+hold, 'close'] - df.loc[i, 'close'])
        if not pnl: continue
        win  = sum(p>0 for p in pnl)
        pf   = (sum(p for p in pnl if p>0) /
                abs(sum(p for p in pnl if p<0))) if any(p<0 for p in pnl) else 99
        results.append((thr, hold, len(pnl), win/len(pnl), pf))
    res = pd.DataFrame(results, columns=["thr","hold","trades","winrate","pf"])
    return res.sort_values("pf", ascending=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    thr_list  = np.arange(0.1, 0.9, 0.05)
    hold_list = [4, 8, 12, 24]
    tbl = run_grid(Path(args.bars), Path(args.model), thr_list, hold_list)
    print(tbl.head(10))
