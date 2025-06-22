#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
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


#%% Confusion Matrix 시각화 및 저장 함수  
def plot_and_save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))

    # 최희영님이 원하시는 기존의 파란색 계열 팔레트 유지
    cmap = sns.light_palette("#3d4e62", as_cmap=True)

    ax = sns.heatmap(
        cm,
        annot=False,  # seaborn의 기본 annotation을 끕니다. (우리가 수동으로 추가할 것이기 때문)
        fmt='g',
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="#a0a0a0"
    )

    # --- 셀 안의 숫자 텍스트 색상을 배경색에 따라 자동으로 변경하는 로직 ---
    # 컬러맵의 최소/최대 값을 기준으로 정규화
    vmin = cm.min()
    vmax = cm.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            # 해당 셀의 배경색을 컬러맵에서 가져옵니다 (RGBA 값).
            bg_color_rgba = cmap(norm(cell_value))

            # 배경색의 밝기(휘도)를 계산합니다. (밝기 계산 공식)
            # RGB 값은 0-1 범위입니다.
            r, g, b, _ = bg_color_rgba 
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)

            # 밝기 임계값을 기준으로 텍스트 색상을 결정합니다.
            # 0.5보다 밝으면 어두운 텍스트, 0.5보다 어두우면 밝은 텍스트(흰색)
            text_color = 'white' if luminance < 0.5 else '#3d4e62' # 최희영님이 원하시는 텍스트 색상 사용

            # 셀에 텍스트 추가 (정확히 중앙에 오도록 ha='center', va='center')
            ax.text(j + 0.5, i + 0.5, f'{cell_value}',
                    ha='center', va='center', color=text_color, fontsize=14, weight='normal')
    # -----------------------------------------------------------------

    ax.set_title(title, fontsize=16, color="#3d4e62", pad=20)
    ax.set_xlabel('Predicted label', fontsize=12, color="#556677")
    ax.set_ylabel('True label', fontsize=12, color="#556677")

    ax.set_xticklabels(['No Depression', 'Depression'], rotation=0)
    ax.set_yticklabels(['No Depression', 'Depression'], rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_and_save_cm(y_test_students, y_pred_students, 'Confusion Matrix - Students', '1.modeling/cm_rf_students.png')
plot_and_save_cm(y_test_professionals, y_pred_professionals, 'Confusion Matrix - Professionals', '1.modeling/cm_rf_professionals.png')
