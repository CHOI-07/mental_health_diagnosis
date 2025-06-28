#%% [1] 라이브러리 로드
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()  # 시각화 저장을 위한 인터랙션 비활성화

#%% [2] 전처리 데이터 로드
path = r"C:\Users\hhhey\Desktop\ME\1.re_project\정신건강_자가진단\0.preprocessing\mental_train_preprocessed.csv"
df = pd.read_csv(path, na_values=[""])

X = df.drop(columns=["Depression", "Name"])
y = df["Depression"]

# 전체 get_dummies 적용 → 컬럼 정렬 맞추기 위함
X_encoded = pd.get_dummies(X, drop_first=True)

# 모델 학습 시 사용한 컬럼 순서 로드
with open("3.shap_analysis/feature_columns.txt", "r") as f:
    feature_cols = f.read().splitlines()

X_encoded = X_encoded[feature_cols]

# 직군 분리
X_students = X_encoded[df["Working Professional or Student"] == 1]
X_workers = X_encoded[df["Working Professional or Student"] == 2]

#%% [3] 모델 로드
xgb_students = joblib.load("3.shap_analysis/xgb_students_model.pkl")
xgb_workers = joblib.load("3.shap_analysis/xgb_professionals_model.pkl")

#%% [4] SHAP 분석 - 학생 그룹
explainer_students = shap.Explainer(xgb_students)
shap_values_students = explainer_students(X_students)

shap.summary_plot(shap_values_students, X_students, show=False)
plt.savefig("3.shap_analysis/shap_summary_students.png", bbox_inches='tight')
plt.clf()

shap.plots.bar(shap_values_students, show=False)
plt.savefig("3.shap_analysis/shap_bar_students.png", bbox_inches='tight')
plt.clf()

#%% [5] SHAP 분석 - 직장인 그룹
explainer_workers = shap.Explainer(xgb_workers)
shap_values_workers = explainer_workers(X_workers)

shap.summary_plot(shap_values_workers, X_workers, show=False)
plt.savefig("3.shap_analysis/shap_summary_professionals.png", bbox_inches='tight')
plt.clf()

shap.plots.bar(shap_values_workers, show=False)
plt.savefig("3.shap_analysis/shap_bar_professionals.png", bbox_inches='tight')
plt.clf()
