## [SHAP 분석 실패 로그]

- 실행 스크립트: `1_shap_analysis.py`
- 에러 메시지 요약:
  - `KeyError: [컬럼명들] not in index`
  - `XGBoostError: shape mismatch (585921 vs. 7923884)`
- 원인:
  - 그룹 분리 후 get_dummies → 컬럼 수 불일치
- 조치:
  - 전체 데이터에서 get_dummies → 이후 그룹 분리
  - 모델 재학습 스크립트로 전환 (`1_model_retrain.py`)


>> 코드
#%% [1] 라이브러리 불러오기
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

#%% [2] 데이터 로드 및 분리
path = r"C:\Users\hhhey\Desktop\ME\1.re_project\정신건강_자가진단\0.preprocessing\mental_train_preprocessed.csv"
df = pd.read_csv(path, na_values=[""])

X = df.drop(columns=["Depression"])
y = df["Depression"]

X_students = X[df["Working Professional or Student"] == 1]
X_workers = X[df["Working Professional or Student"] == 2]

X_students = pd.get_dummies(X_students, drop_first=True)
X_workers = pd.get_dummies(X_workers, drop_first=True)

#%% [3] 모델 불러오기
xgb_students = joblib.load("1.modeling/xgb_students_model.pkl")
xgb_workers = joblib.load("1.modeling/xgb_professionals_model.pkl")

model_columns = xgb_students.get_booster().feature_names

#%% [4] 피처 정렬 및 정합성 맞춤
X_students, X_workers = X_students.align(X_workers, join='outer', axis=1, fill_value=0)
X_students = X_students[model_columns]
X_workers = X_workers[model_columns]

#%% [5] SHAP 분석 (학생)
explainer_students = shap.Explainer(xgb_students)
shap_values_students = explainer_students(X_students)

shap.summary_plot(shap_values_students, X_students, show=False)
plt.savefig("3.shap_analysis/shap_summary_students.png", bbox_inches='tight')
plt.clf()

shap.plots.bar(shap_values_students, show=False)
plt.savefig("3.shap_analysis/shap_bar_students.png", bbox_inches='tight')
plt.clf()

#%% [6] SHAP 분석 (직장인)
explainer_workers = shap.Explainer(xgb_workers)
shap_values_workers = explainer_workers(X_workers)

shap.summary_plot(shap_values_workers, X_workers, show=False)
plt.savefig("3.shap_analysis/shap_summary_professionals.png", bbox_inches='tight')
plt.clf()

shap.plots.bar(shap_values_workers, show=False)
plt.savefig("3.shap_analysis/shap_bar_professionals.png", bbox_inches='tight')
plt.clf()
