# #%% 라이브러리
# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from xgboost import XGBClassifier
# from pathlib import Path
# import joblib

# # 데이터 로드
# base_dir = Path(__file__).resolve().parent.parent
# data_path = base_dir / '0.preprocessing' / 'mental_train_preprocessed.csv'
# df = pd.read_csv(data_path)

# # 그룹별 분리 (원본 컬럼 기준)
# df_students = df[df['Working Professional or Student'] == 1].copy()
# df_workers = df[df['Working Professional or Student'].isin([2, 3])].copy()

# drop_cols = ['Depression', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']
# target = 'Depression'

# # Total Feature 생성
# for d in [df_students, df_workers]:
#     d['Total_Pressure'] = d[['Academic Pressure', 'Work Pressure']].sum(axis=1)
#     d['Total_Satisfaction'] = d[['Study Satisfaction', 'Job Satisfaction']].sum(axis=1)

# # 공통 학습 함수
# def train_and_eval(df, group_name):
#     X = df.drop(columns=drop_cols)
#     y = df[target]
#     X = pd.get_dummies(X, drop_first=True)

#     model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
#     model.fit(X, y)
#     y_pred = model.predict(X)

#     cm = confusion_matrix(y, y_pred)
#     report = classification_report(y, y_pred, target_names=['No Depression', 'Depression'])

#     print(f"\n===== XGBoost - {group_name} Classification Report =====")
#     print(report)

#     # 모델 저장
#     model_path = base_dir / f'1.modeling/xgb_{group_name.lower()}_model.pkl'
#     joblib.dump(model, model_path)

#     # 리포트 저장
#     report_path = base_dir / f'1.modeling/xgb_{group_name.lower()}_report.txt'
#     with open(report_path, 'w') as f:
#         f.write(f"XGBoost - {group_name} Classification Report\n")
#         f.write(report)

#     # Confusion Matrix 저장
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No Depression', 'Depression'],
#                 yticklabels=['No Depression', 'Depression'])
#     plt.title(f'Confusion Matrix - {group_name} (XGBoost)')
#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")
#     plt.tight_layout()
#     fig_path = base_dir / f'2.visualization/cm_xgb_{group_name.lower()}.png'
#     plt.savefig(fig_path, dpi=300)
#     plt.close()

# # 실행
# train_and_eval(df_students, "Students")
# train_and_eval(df_workers, "Professionals")
#%% 라이브러리
import matplotlib.pyplot as plt
import pandas as pd

#데이터 정의 (로지스틱 회귀 기준 성능)
data = {
    '학생': [0.88, 0.83, 0.86],  
    '직장인': [0.75, 0.98, 0.96],
    # '전체': [0.0, 0.90, 0.82]
}
index = ['F1-score (Depression)', 'F1-score (No Depression)', 'Accuracy']
df = pd.DataFrame(data, index=index)

# 폰트 및 색상 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Windows는 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
colors = ['#E19992', '#D54344', '#a0a0a0']  # 학생 / 직장인 / 전체

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
bars = df.plot(kind='bar', ax=ax, width=0.6, color=colors)

# 수치 라벨 추가
for container in bars.containers:
    for bar in container:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='#3d4e62')

# 타이틀 및 축 설정
ax.set_title('XGBoost 성능 비교 (학생 vs 직장인)', fontsize=16, color='#3d4e62', pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0.6, 1.05)
ax.set_xticklabels(df.index, rotation=0, fontsize=11)
ax.legend(title='그룹', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# 저장
plt.savefig('1.modeling/XGBoost_group_comparison_final.png', dpi=300)
plt.show()
# %%
