# -*- coding: utf-8 -*-
"""
ファイル名: step4_random_search_model.py

ステップ4: ランダムサーチによるハイパーパラメータ探索
  1. step2_time_features.csv の読み込み
  2. 特徴量とラベルの分割
  3. 学習データ/テストデータ分割（70%/30%）
  4. RandomizedSearchCV で探索空間を設定し試行（n_iter=20）
  5. 最適モデルでテストセットを評価（accuracy, precision, recall, f1）
  6. 特徴量重要度を表示（上位20）
  7. ベストモデルを random_model.pkl として保存

使用方法: python step4_random_search_model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import lightgbm as lgb
import joblib
from scipy.stats import uniform, randint

# 1. ファイルパス設定
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, 'step2_time_features.csv')
model_file = os.path.join(BASE_DIR, 'random_model.pkl')

# 2. データ読み込み
df = pd.read_csv(input_file)

# 3. 特徴量とラベルの分割
y = df['label']
X = df.drop(columns=['entry_time', 'bar_time', 'exit_time', 'label', 'entry_price', 'exit_price'])

print("学習に使用された特徴量の列名:", X.columns.tolist())
# 4. 学習データ/テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. 探索空間設定
param_dist = {
    'learning_rate': uniform(0.01, 0.19),
    'n_estimators': randint(100, 500),
    'num_leaves': randint(20, 100),
    'max_depth': randint(5, 20),
    'feature_fraction': uniform(0.6, 0.4),
    'bagging_fraction': uniform(0.6, 0.4),
    'min_child_samples': randint(5, 30),
    'min_split_gain': uniform(0.0, 0.5)
}
base_clf = lgb.LGBMClassifier(random_state=42)

# 6. RandomizedSearchCV 実行
rand_search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
print("RandomizedSearchCV を実行中... 20 回の試行で最適パラメータを検索します")
rand_search.fit(X_train, y_train)
print(f"Best params: {rand_search.best_params_}")
print(f"Best CV F1 Score: {rand_search.best_score_:.4f}\n")

# 7. 最適モデルでテスト評価
best_clf = rand_search.best_estimator_
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

# 8. 特徴量重要度表示
importances = best_clf.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
print("=== 特徴量重要度 (Top 20) ===")
print(feat_imp.to_string(index=False))

# 9. モデル保存
joblib.dump(best_clf, model_file)
print(f"ベストモデルを保存しました: {model_file}")
