#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#%%
# 데이터 경로 설정 및 로드
path = '0.preprocessing/mental_train_preprocessed.csv'
df = pd.read_csv(path, na_values=['', np.nan])

#%%
# 학생(1), 직장인(2,3) 데이터 분리
students_df = df[df['Working Professional or Student'] == 1]
professionals_df = df[df['Working Professional or Student'].isin([2, 3])]

target_variable = 'Depression'

# 모델에 넣지 않을 컬럼 목록
features_to_drop = [target_variable, 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']

# 입력(X), 타겟(y) 분리
X_students = students_df.drop(columns=features_to_drop)
y_students = students_df[target_variable]

X_professionals = professionals_df.drop(columns=features_to_drop)
y_professionals = professionals_df[target_variable]

#%%
# One-Hot Encoding 적용 (범주형 변수)
X_students = pd.get_dummies(X_students, drop_first=True)
X_professionals = pd.get_dummies(X_professionals, drop_first=True)

#%%
# 학습/검증 데이터 분리
X_train_students, X_test_students, y_train_students, y_test_students = train_test_split(
    X_students, y_students, test_size=0.2, random_state=42)

X_train_professionals, X_test_professionals, y_train_professionals, y_test_professionals = train_test_split(
    X_professionals, y_professionals, test_size=0.2, random_state=42)

#%%
# 랜덤포레스트 모델 학습
rf_students = RandomForestClassifier(random_state=42, n_estimators=100)
rf_students.fit(X_train_students, y_train_students)

rf_professionals = RandomForestClassifier(random_state=42, n_estimators=100)
rf_professionals.fit(X_train_professionals, y_train_professionals)

#%%
# 예측 수행
y_pred_students = rf_students.predict(X_test_students)
y_pred_professionals = rf_professionals.predict(X_test_professionals)

#%%
# 평가 결과 출력 및 텍스트 저장 함수
target_names = ['No Depression', 'Depression']

def save_classification_report(y_true, y_pred, filename, title, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(f"===== {title} =====")
    print(report)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write(report)

save_classification_report(y_test_students, y_pred_students, '1.modeling/rf_students_report.txt', 
                           'RandomForest - Students Classification Report', target_names)
save_classification_report(y_test_professionals, y_pred_professionals, '1.modeling/rf_professionals_report.txt', 
                           'RandomForest - Professionals Classification Report', target_names)

#%%
# Confusion Matrix 시각화 및 저장 함수
def plot_and_save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_and_save_cm(y_test_students, y_pred_students, 'Confusion Matrix - Students', '1.modeling/cm_students.png')
plot_and_save_cm(y_test_professionals, y_pred_professionals, 'Confusion Matrix - Professionals', '1.modeling/cm_professionals.png')
