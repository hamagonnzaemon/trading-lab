import pandas as pd
import numpy as np

# 評価指標
from sklearn.model_selection import train_test_split
from sklearn.metrics      import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb

# 1. データ読み込み
df = pd.read_csv("trade_log/trades_ml_ready.csv")
print("読み込み完了：", df.shape)
print(df.head())
print(df["label"].value_counts(), "\n")  # 1=TP,0=SL のバランスを見る

# 2. 欠損チェック
print("欠損値サマリ：\n", df.isna().sum(), "\n")

# 3. 欠損処理
#    ここではシンプルに「欠損行を落とす」
df = df.dropna().reset_index(drop=True)
print("欠損除去後：", df.shape, "\n")

# 4. 特徴量とラベル分離
y = df["label"].astype(int)       # 1/0
X = df.drop(columns=["entry_time","exit_time","label","bar_time"])

# 5. 学習用・検証用に分割
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape)

# 6. LightGBMによる分類モデル
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    early_stopping_rounds=50,
    verbose=50
)

# 7. 評価
y_pred = model.predict(X_val)
y_proba= model.predict_proba(X_val)[:,1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print("AUC     :", roc_auc_score(y_val, y_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
