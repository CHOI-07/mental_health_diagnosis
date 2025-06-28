#%% 라이브러리
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

#%% 데이터 불러오기
path = '0.preprocessing/mental_train_preprocessed.csv'
df = pd.read_csv(path, na_values=['', np.nan])

#%% 학생, 직장인 분리
students_df = df[df['Working Professional or Student'] == 1]
professionals_df = df[df['Working Professional or Student'].isin([2, 3])]

target = 'Depression'
features_to_drop = [target, 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']

# 학생 데이터
X_students = students_df.drop(columns=features_to_drop)
y_students = students_df[target]
X_students = pd.get_dummies(X_students, drop_first=True)

# 직장인 데이터
X_professionals = professionals_df.drop(columns=features_to_drop)
y_professionals = professionals_df[target]
X_professionals = pd.get_dummies(X_professionals, drop_first=True)

#%% 학습/검증 데이터 분리
X_train_students, X_test_students, y_train_students, y_test_students = train_test_split(
    X_students, y_students, test_size=0.2, random_state=42)

X_train_professionals, X_test_professionals, y_train_professionals, y_test_professionals = train_test_split(
    X_professionals, y_professionals, test_size=0.2, random_state=42)

#%% 모델 생성 및 학습
model_students = LogisticRegression(max_iter=1000, random_state=42)
model_students.fit(X_train_students, y_train_students)

model_professionals = LogisticRegression(max_iter=1000, random_state=42)
model_professionals.fit(X_train_professionals, y_train_professionals)

#%% 예측
y_pred_students = model_students.predict(X_test_students)
y_pred_professionals = model_professionals.predict(X_test_professionals)

#%% 평가 및 리포트 저장 함수
def save_classification_report(y_true, y_pred, filename, title):
    target_names = ['No Depression', 'Depression']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(f"===== {title} =====")
    print(report)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write(report)

save_classification_report(y_test_students, y_pred_students, '1.modeling/logreg_students_report.txt', 'Logistic Regression - Students Classification Report')
save_classification_report(y_test_professionals, y_pred_professionals, '1.modeling/logreg_professionals_report.txt', 'Logistic Regression - Professionals Classification Report')

#%% 혼동 행렬 시각화 및 저장 함수
def plot_and_save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))

    cmap = sns.light_palette("#3d4e62", as_cmap=True)

    ax = sns.heatmap(
        cm,
        annot=False,                 # seaborn 기본 annotation 비활성화 (직접 텍스트 추가)
        fmt='g',
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="#a0a0a0"
    )

    # 셀 배경색 밝기에 따른 텍스트 색상 자동 조절
    vmin = cm.min()
    vmax = cm.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            bg_color_rgba = cmap(norm(cell_value))
            r, g, b, _ = bg_color_rgba
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'white' if luminance < 0.5 else '#3d4e62'
            ax.text(j + 0.5, i + 0.5, f'{cell_value}', ha='center', va='center', color=text_color, fontsize=14, weight='normal')

    ax.set_title(title, fontsize=16, color="#3d4e62", pad=20)
    ax.set_xlabel('Predicted label', fontsize=12, color="#556677")
    ax.set_ylabel('True label', fontsize=12, color="#556677")

    ax.set_xticklabels(['No Depression', 'Depression'], rotation=0)
    ax.set_yticklabels(['No Depression', 'Depression'], rotation=0)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


plot_and_save_cm(y_test_students, y_pred_students, 'Confusion Matrix - Students', '1.modeling/cm_logreg_students.png')
plot_and_save_cm(y_test_professionals, y_pred_professionals, 'Confusion Matrix - Professionals', '1.modeling/cm_logreg_professionals.png')
