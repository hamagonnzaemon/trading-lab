# -*- coding: utf-8 -*-
"""
ファイル名: step3_baseline_model.py

ステップ3: LightGBM によるベースライン学習・評価（パターン①: GridSearchCV + 特徴量重要度可視化）
  1. step2_time_features.csv の読み込み
  2. 特徴量とラベルの分割
  3. 学習データ/テストデータ分割（70%/30%）
  4. GridSearchCV でハイパーパラメータ探索
  5. 最適モデルでテストセットを評価（accuracy, precision, recall, f1）
  6. 特徴量重要度を表示（上位20）
  7. ベストモデルを baseline_model.pkl として保存

使用方法: python step3_baseline_model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import lightgbm as lgb
import joblib

# 1. ファイルパス設定
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, 'step2_time_features.csv')
model_file = os.path.join(BASE_DIR, 'baseline_model.pkl')

# 2. データ読み込み
df = pd.read_csv(input_file)

# 3. 特徴量とラベルの分割
y = df['label']
X = df.drop(columns=['entry_time', 'bar_time', 'exit_time', 'label', 'entry_price', 'exit_price'])

# 4. 学習データ/テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. GridSearchCV による探索
tuned_params = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'num_leaves': [31, 64],
    'max_depth': [-1, 10]
}
base_clf = lgb.LGBMClassifier(
    random_state=42,
    min_child_samples=10,
    min_split_gain=0.0
)
grid = GridSearchCV(
    estimator=base_clf,
    param_grid=tuned_params,
    scoring='f1',
    cv=2,
    n_jobs=-1,
    verbose=1
)
print("GridSearchCV を実行中... 最適パラメータを検索します")
grid.fit(X_train, y_train)

# 最適パラメータとスコアを表示
print(f"Best params: {grid.best_params_}")
print(f"Best CV F1 Score: {grid.best_score_:.4f}\n")

# 最適モデルを取得
best_clf = grid.best_estimator_

# 6. テストセットで予測・評価
y_pred = best_clf.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print("=== モデル評価（テストセット）===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}\n")
print("=== 詳細レポート ===")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. 特徴量重要度を表示（上位20）
importances = best_clf.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
print("=== 特徴量重要度 (Top 20) ===")
print(feat_imp.to_string(index=False))

# 8. ベストモデルを保存
joblib.dump(best_clf, model_file)
print(f"ベストモデルを保存しました: {model_file}")
