#%% 라이브러리
import matplotlib.pyplot as plt
import pandas as pd

#%% 데이터 정의 (로지스틱 회귀 기준 성능)
data = {
    '학생': [0.85, 0.79, 0.83],
    '직장인': [0.71, 0.98, 0.96],
    # '전체': [0.0, 0.90, 0.82]
}
index = ['F1-score (Depression)', 'F1-score (No Depression)', 'Accuracy']
df = pd.DataFrame(data, index=index)

#%% 폰트 및 색상 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Windows는 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
colors = ['#E19992', '#D54344', '#a0a0a0']  # 학생 / 직장인 / 전체

#%% 시각화
fig, ax = plt.subplots(figsize=(10, 6))
bars = df.plot(kind='bar', ax=ax, width=0.6, color=colors)

# 수치 라벨 추가
for container in bars.containers:
    for bar in container:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='#3d4e62')

# 타이틀 및 축 설정
ax.set_title('Logistic Regression 성능 비교 (학생 vs 직장인 vs 전체)', fontsize=16, color='#3d4e62', pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0.6, 1.05)
ax.set_xticklabels(df.index, rotation=0, fontsize=11)
ax.legend(title='그룹', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

#%% 저장
plt.savefig('1.modeling/logreg_group_comparison_final.png', dpi=300)
plt.show()

#-----------------------
# #%% 전체 데이터 성능 비교 
# #%% 전체 데이터 기준 Logistic 회귀 평가 및 저장
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from pathlib import Path
# import numpy as np

# # 현재 스크립트 기준 루트 경로
# base_dir = Path('/Users/hee/Desktop/ME/1.re_project/정신건강_자가진단')
# data_path = base_dir / '0.preprocessing' / 'mental_train_preprocessed.csv'

# # 데이터 로드
# df = pd.read_csv(data_path, na_values=['', np.nan])

# # Feature/Target 설정
# df['Total_Pressure'] = df[['Academic Pressure', 'Work Pressure']].sum(axis=1)
# df['Total_Satisfaction'] = df[['Study Satisfaction', 'Job Satisfaction']].sum(axis=1)


# drop_cols = ['Depression', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']
# target = 'Depression'

# X = df.drop(columns=drop_cols)
# y = df[target]

# # One-hot Encoding
# X = pd.get_dummies(X, drop_first=True)

# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 모델 학습
# logreg = LogisticRegression(max_iter=1000, solver='saga', random_state=42, n_jobs=-1)
# logreg.fit(X_train, y_train)

# # 예측
# y_pred = logreg.predict(X_test)

# # 리포트 생성
# target_names = ['No Depression', 'Depression']
# report_text = classification_report(y_test, y_pred, target_names=target_names)
# accuracy = accuracy_score(y_test, y_pred)

# # 출력
# print("\n[전체 데이터 기준] Logistic Regression 성능 리포트\n")
# print(report_text)
# print(f"정확도: {accuracy:.3f}")

# # 리포트 저장
# with open('1.modeling/logreg_all_report.txt', 'w', encoding='utf-8') as f:
#     f.write("[전체 데이터 기준] Logistic Regression 성능 리포트\n\n")
#     f.write(report_text)
#     f.write(f"\n정확도: {accuracy:.3f}\n")


