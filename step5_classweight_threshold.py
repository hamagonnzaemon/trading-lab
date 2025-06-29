# -*- coding: utf-8 -*-
"""
ファイル名: step5_classweight_threshold.py

ステップ5: クラス不均衡対策 & F1最適閾値チューニング
  1. step2_time_features.csv の読み込み
  2. 特徴量とラベルの分割
  3. 学習データ/テストデータ分割（70%/30%）
  4. class_weight='balanced' を適用した LightGBM で学習
  5. テストセットで確率出力 → F1 が最大となる閾値を探索
  6. 最適閾値で再予測し、評価指標を表示
  7. ベストモデルと閾値を classweight_model.pkl / best_threshold.txt として保存

使用方法: python step5_classweight_threshold.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, accuracy_score, precision_score, recall_score
import lightgbm as lgb
import joblib

# 1. パス設定
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, 'step2_time_features.csv')
model_file = os.path.join(BASE_DIR, 'classweight_model.pkl')
threshold_file = os.path.join(BASE_DIR, 'best_threshold.txt')

# 2. データロード
df = pd.read_csv(input_file)

# 3. 特徴量・ラベル分割
y = df['label']
X = df.drop(columns=['entry_time', 'bar_time', 'exit_time', 'label', 'entry_price', 'exit_price'])
print("学習に使用された特徴量の列名:", X.columns.tolist())

# 4. train/test 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. class_weight='balanced' で学習
clf = lgb.LGBMClassifier(
    random_state=42,
    class_weight='balanced',
    learning_rate=0.05,
    n_estimators=300,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=10,
    min_split_gain=0.0,
)
clf.fit(X_train, y_train)

# 6. 予測確率取得
proba = clf.predict_proba(X_test)[:, 1]

# 7. Precision-Recall 曲線から F1 最大の閾値を探索
prec, rec, thresh = precision_recall_curve(y_test, proba)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresh[best_idx]
print(f"最適閾値: {best_threshold:.4f}  (F1={f1_scores[best_idx]:.4f})")

# 8. 閾値適用してラベル化
y_pred = (proba >= best_threshold).astype(int)

# 9. 指標計算
acc  = accuracy_score(y_test, y_pred)
prec_m = precision_score(y_test, y_pred, zero_division=0)
rec_m  = recall_score(y_test, y_pred, zero_division=0)
f1_m   = f1_score(y_test, y_pred, zero_division=0)
print("=== 最適閾値適用後の評価 ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec_m:.4f}")
print(f"Recall   : {rec_m:.4f}")
print(f"F1 Score : {f1_m:.4f}\n")
print("=== 詳細レポート ===")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. モデルと閾値を保存
joblib.dump(clf, model_file)
with open(threshold_file, 'w') as f:
    f.write(str(best_threshold))
print(f"モデル保存: {model_file}")
print(f"閾値保存: {threshold_file}")
