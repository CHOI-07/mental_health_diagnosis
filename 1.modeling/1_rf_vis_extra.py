import matplotlib.pyplot as plt
import pandas as pd

data = {
    '학생': [0.85, 0.79, 0.83],
    '직장인': [0.71, 0.98, 0.96],
    '전체': [0.69, 0.94, 0.91]  # Depression F1, No Depression F1, Accuracy
}
index = ['F1-score (Depression)', 'F1-score (No Depression)', 'Accuracy']
df = pd.DataFrame(data, index=index)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

colors = ['#E19992', '#D54344', '#a0a0a0']
ax = df.plot(kind='bar', figsize=(9, 5), width=0.6, color=colors)
ax.set_title('RandomForest 성능 비교 (학생 vs 직장인)', fontsize=15, color='#3d4e62', pad=15)
ax.set_ylabel('Score')
ax.set_ylim(0.6, 1.0)
ax.set_xticklabels(df.index, rotation=0)

for p in ax.patches:
    value = p.get_height()
    ax.annotate(f'{value:.2f}', (p.get_x() + p.get_width() / 2, value + 0.01),
                ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('1.modeling/rf_group_comparison.png', dpi=300)
plt.show()

# #%% 전체 데이터 기준 RandomForest 평가 및 저장
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from pathlib import Path
# import numpy as np

# # 경로 설정
# base_dir = Path('/Users/hee/Desktop/ME/1.re_project/정신건강_자가진단')
# data_path = base_dir / '0.preprocessing' / 'mental_train_preprocessed.csv'

# # 데이터 로드
# df = pd.read_csv(data_path, na_values=['', np.nan])

# # 파생변수 생성
# df['Total_Pressure'] = df[['Academic Pressure', 'Work Pressure']].sum(axis=1)
# df['Total_Satisfaction'] = df[['Study Satisfaction', 'Job Satisfaction']].sum(axis=1)

# # Feature/Target 설정
# drop_cols = ['Depression', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']
# target = 'Depression'

# X = df.drop(columns=drop_cols)
# y = df[target]

# # 인코딩
# X = pd.get_dummies(X, drop_first=True)

# # 학습/테스트 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 모델 학습
# rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
# rf.fit(X_train, y_train)

# # 예측
# y_pred = rf.predict(X_test)

# # 평가
# target_names = ['No Depression', 'Depression']
# report_text = classification_report(y_test, y_pred, target_names=target_names)
# accuracy = accuracy_score(y_test, y_pred)

# # 출력
# print("\n[전체 데이터 기준] RandomForest 성능 리포트\n")
# print(report_text)
# print(f"정확도: {accuracy:.3f}")

# # 저장
# save_path = base_dir / '1.modeling' / 'rf_all_report.txt'
# with open(save_path, 'w', encoding='utf-8') as f:
#     f.write("[전체 데이터 기준] RandomForest 성능 리포트\n\n")
#     f.write(report_text)
#     f.write(f"\n정확도: {accuracy:.3f}\n")
