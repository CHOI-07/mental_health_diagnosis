#%% [1] 라이브러리 불러오기
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#%% [2] 데이터 불러오기
path = r"C:\Users\hhhey\Desktop\ME\1.re_project\정신건강_자가진단\0.preprocessing\mental_train_preprocessed.csv"
df = pd.read_csv(path, na_values=[""])


#%% [3] 타겟/피처 분리 및 그룹화
X = df.drop(columns=["Depression", "Name"])
y = df["Depression"]
group = df["Working Professional or Student"]

# 전체 데이터에 get_dummies 일괄 적용
X_encoded = pd.get_dummies(X, drop_first=True)

# 컬럼 정렬 기준 저장
feature_columns = X_encoded.columns

# 그룹별로 나누기
X_students = X_encoded[group == 1]
y_students = y[group == 1]

X_workers = X_encoded[group == 2]
y_workers = y[group == 2]

#%% [4] 모델 학습 함수 정의
def train_and_save_model(X, y, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

#%% [5] 학생 모델 저장
train_and_save_model(X_students, y_students, "3.shap_analysis/xgb_students_model.pkl")

#%% [6] 직장인 모델 저장
train_and_save_model(X_workers, y_workers, "3.shap_analysis/xgb_professionals_model.pkl")
#%% [7] 피처 목록 저장 (추후 shap에서 로딩용)
with open("3.shap_analysis/feature_columns.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(feature_columns))

print("✅ 전체 재학습 완료")
