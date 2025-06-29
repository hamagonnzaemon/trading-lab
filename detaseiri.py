import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from pathlib import Path

# ▼ 1. データ読込
df = pd.read_csv(Path.home() / "Desktop/trade_log/ml_dataset.csv")

# ▼ 2. 欠損値チェック（NaN があれば 0 で埋める or 直前値で埋める）
print(df.isna().sum())               # ざっと確認
df = df.fillna(0)                    # ここでは簡易に 0 埋め

# ▼ 3. 説明変数 / 目的変数
X = df.drop(columns=["y"])
y = df["y"].astype(int)              # ラベルを int に
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=0
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    pred = model.predict(X.iloc[test_idx])
    acc  = accuracy_score(y.iloc[test_idx], pred)
    auc  = roc_auc_score(y.iloc[test_idx], pred)
    scores.append(acc)
    
    print(f"Fold {fold}: ACC={acc:.3f}  AUC={auc:.3f}")

print(f"\nBaseline ACC  avg={sum(scores)/len(scores):.3f}")
