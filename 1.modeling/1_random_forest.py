#%% 라이브러리
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#%% 데이터 로드
path = '0.preprocessing/mental_train_preprocessed.csv'
df = pd.read_csv(path)

#%% 학생/직장인 분리
students_df = df[df['Working Professional or Student'] == 1]
professionals_df = df[df['Working Professional or Student'].isin([2, 3])]

target_variable = 'Depression'
features_to_drop = [target_variable, 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']

X_students = students_df.drop(columns=features_to_drop)
y_students = students_df[target_variable]
X_professionals = professionals_df.drop(columns=features_to_drop)
y_professionals = professionals_df[target_variable]

#%% One-Hot Encoding
X_students = pd.get_dummies(X_students, drop_first=True)
X_professionals = pd.get_dummies(X_professionals, drop_first=True)

#%% 컬럼 일치 강제 정렬
all_student_cols = list(set(X_students.columns))
X_students = X_students.reindex(columns=all_student_cols, fill_value=0)
X_professionals = X_professionals.reindex(columns=all_student_cols, fill_value=0)

#%% 데이터 분리
X_train_students, X_test_students, y_train_students, y_test_students = train_test_split(
    X_students, y_students, test_size=0.2, random_state=42)

X_train_professionals, X_test_professionals, y_train_professionals, y_test_professionals = train_test_split(
    X_professionals, y_professionals, test_size=0.2, random_state=42)

#%% 모델 학습
rf_students = RandomForestClassifier(random_state=42, n_estimators=100)
rf_students.fit(X_train_students, y_train_students)

rf_professionals = RandomForestClassifier(random_state=42, n_estimators=100)
rf_professionals.fit(X_train_professionals, y_train_professionals)

#%% 예측
y_pred_students = rf_students.predict(X_test_students)
y_pred_professionals = rf_professionals.predict(X_test_professionals)

#%% 리포트 저장
from sklearn.metrics import classification_report

def save_classification_report(y_true, y_pred, filename, title, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(f"\n===== {title} =====\n{report}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write(report)

target_names = ['No Depression', 'Depression']
save_classification_report(y_test_students, y_pred_students,
    '1.modeling/rf_students_report.txt', 'RandomForest - Students', target_names)
save_classification_report(y_test_professionals, y_pred_professionals,
    '1.modeling/rf_professionals_report.txt', 'RandomForest - Professionals', target_names)

#%% Confusion Matrix 저장
def plot_and_save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cmap = sns.light_palette("#3d4e62", as_cmap=True)
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(cm, annot=False, fmt='g', cmap=cmap, cbar=False,
                     linewidths=0.5, linecolor="#a0a0a0")
    vmin, vmax = cm.min(), cm.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            bg = cmap(norm(val))
            luminance = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
            text_color = 'white' if luminance < 0.5 else '#3d4e62'
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha='center', va='center', color=text_color, fontsize=14)
    ax.set_title(title, fontsize=16, color="#3d4e62", pad=20)
    ax.set_xlabel('Predicted', fontsize=12, color="#556677")
    ax.set_ylabel('True', fontsize=12, color="#556677")
    ax.set_xticklabels(['No Depression', 'Depression'], rotation=0)
    ax.set_yticklabels(['No Depression', 'Depression'], rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_and_save_cm(y_test_students, y_pred_students,
    'Confusion Matrix - Students', '1.modeling/cm_rf_students.png')
plot_and_save_cm(y_test_professionals, y_pred_professionals,
    'Confusion Matrix - Professionals', '1.modeling/cm_rf_professionals.png')

#%% Feature Importance 저장
def save_feature_importance(model, X, filename):
    importances = model.feature_importances_
    feature_names = X.columns
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    fi_df.to_csv(filename, index=False)

save_feature_importance(rf_students, X_train_students, '1.modeling/rf_students_feature_importance.csv')
save_feature_importance(rf_professionals, X_train_professionals, '1.modeling/rf_professionals_feature_importance.csv')
