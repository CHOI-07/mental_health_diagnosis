#%% SHAP Top 5 변수 중요도 비교 시각화 (학생 vs 직장인)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'  # macOS용

plt.ioff()

# 모델 및 데이터 로드
xgb_students = joblib.load("3.shap_analysis/xgb_students_model.pkl")
xgb_workers = joblib.load("3.shap_analysis/xgb_professionals_model.pkl")

df = pd.read_csv("0.preprocessing/mental_train_preprocessed.csv")
X = df.drop(columns=["Depression", "Name"])
y = df["Depression"]

# get_dummies 후 정렬
X_encoded = pd.get_dummies(X, drop_first=True)
with open("3.shap_analysis/feature_columns.txt", "r") as f:
    feature_cols = f.read().splitlines()

# 열 재정렬
X_encoded = X_encoded.reindex(columns=feature_cols, fill_value=0)

# 그룹 분리
X_students = X_encoded[df["Working Professional or Student"] == 1]
X_workers = X_encoded[df["Working Professional or Student"] == 2]

# SHAP 계산 (TreeExplainer + .values)
explainer_students = shap.TreeExplainer(xgb_students)
explainer_workers = shap.TreeExplainer(xgb_workers)

shap_values_students = explainer_students.shap_values(X_students.values)
shap_values_workers = explainer_workers.shap_values(X_workers.values)

# SHAP 중요도 평균 계산
student_importance = np.abs(shap_values_students).mean(axis=0)
worker_importance = np.abs(shap_values_workers).mean(axis=0)

# 중요도 데이터프레임
student_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": student_importance,
    "group": "학생"
})
worker_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": worker_importance,
    "group": "직장인"
})
all_df = pd.concat([student_df, worker_df])

# Top 5 추출
top_features = (
    all_df.groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)
df_top5 = all_df[all_df["feature"].isin(top_features)]

# 시각화
plt.figure(figsize=(8, 5))
bar_labels = []
for i, feature in enumerate(top_features):
    values = df_top5[df_top5["feature"] == feature]
    for _, row in values.iterrows():
        offset = 0 if row["group"] == "학생" else 0.4
        color = "#3d4e62" if row["group"] == "학생" else "#f8dcde"
        label = row["group"]
        if label not in bar_labels:
            plt.barh(i + offset, row["importance"], height=0.35, color=color, label=label)
            bar_labels.append(label)
        else:
            plt.barh(i + offset, row["importance"], height=0.35, color=color)

plt.yticks(np.arange(len(top_features)) + 0.2, labels=top_features)
plt.xlabel("평균 SHAP 중요도")
plt.title("학생 vs 직장인 상위 변수 중요도 비교")
plt.legend()
plt.tight_layout()
plt.savefig("3.shap_analysis/shap_bar_group_compare_top5.png", dpi=300)
plt.clf()
