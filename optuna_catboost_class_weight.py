#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optuna_catboost_class_weight.py
――――――――――――――――――――――――――――――――――――――――――――――――
2 クラス分類 (label=0/1) 用 CatBoost モデルの
class_weight & 主要ハイパーパラメータを Optuna で最適化。
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

#─────────────────────────────
# ① 設定
#─────────────────────────────
DATA_CSV   = Path("step2_time_features.csv")   # ← CSV の場所
TARGET_COL = "label"                           # ← 0 / 1 が入っている列
MODEL_CBM  = Path("best_catboost_model.cbm")
MODEL_PKL  = Path("best_catboost_model.pkl")

N_TRIALS   = 150     # Optuna 試行回数
N_SPLITS   = 5       # 時系列 CV 分割数

#─────────────────────────────
# ② データ読み込み & チェック
#─────────────────────────────
if not DATA_CSV.exists():
    sys.exit(f"★CSV が見つかりません → {DATA_CSV.resolve()}")

df = pd.read_csv(DATA_CSV)

if TARGET_COL not in df.columns:
    print("★TARGET_COL が見つかりません:", TARGET_COL)
    print("▼列名一覧:", list(df.columns))
    sys.exit()

if df.isna().any().any():
    print("★警告: 欠損値を含む行を削除しました。")
    df = df.dropna()

DROP_COLS = ["entry_time", "exit_time", "bar_time"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET_COL])
y = df[TARGET_COL]

print("▼ label 分布:", y.value_counts().to_dict())
print("▼ 特徴量数 :", X.shape[1])

#─────────────────────────────
# ③ 時系列 CV
#─────────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

#─────────────────────────────
# ④ Optuna 目的関数
#─────────────────────────────
def objective(trial):
    # ------ クラス重み ------
    w0 = trial.suggest_float("w0", 0.2, 10.0, log=True)
    w1 = trial.suggest_float("w1", 0.2, 10.0, log=True)

    # ------ CatBoost ハイパーパラメータ ------
    params = {
        "loss_function": "Logloss",
        "eval_metric":   "F1",
        "iterations":    trial.suggest_int("iterations", 300, 1500),
        "depth":         trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "random_state":  42,
        "verbose":       False,
        "class_weights": [w0, w1],
    }

    f1_scores = []
    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_train, y_train),
            eval_set=Pool(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        y_pred = model.predict(Pool(X_valid, y_valid))
        f1_scores.append(f1_score(y_valid, y_pred))

    return -sum(f1_scores) / len(f1_scores)  # Optuna は minimize

#─────────────────────────────
# ⑤ 最適化の実行
#─────────────────────────────
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("\n★ Best params:", study.best_params)
print("★ Best CV F1  :", -study.best_value)

#─────────────────────────────
# ⑥ ベストモデルを全データで学習し直して保存
#─────────────────────────────
best = study.best_params
class_weights = [best.pop("w0"), best.pop("w1")]

best_model = CatBoostClassifier(
    **best,
    loss_function="Logloss",
    eval_metric="F1",
    random_state=42,
    verbose=False,
    class_weights=class_weights,
)
best_model.fit(X, y)

best_model.save_model(MODEL_CBM)
joblib.dump(best_model, MODEL_PKL)

print(f"✓ モデル保存完了 → {MODEL_CBM.name} / {MODEL_PKL.name}")
