#%% 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 한글 폰트 설정 (윈도우용 예시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#%% 경로 설정
base_dir = Path(__file__).parent.parent  # 프로젝트 루트 (2.visualization 폴더 기준 두 단계 위)

raw_data_path = base_dir / 'mental_train.csv'
preprocessed_data_path = base_dir / '0.preprocessing' / 'mental_train_preprocessed.csv'

# 데이터 로드
df_raw = pd.read_csv(raw_data_path, na_values=["", np.nan])
df_clean = pd.read_csv(preprocessed_data_path)

#%% 1. 데이터 상태 비교 (터미널 출력)
print("=== 전처리 전 데이터 info ===")
print(df_raw.info())
print("\n=== 전처리 전 데이터 describe ===")
print(df_raw.describe())

print("\n=== 전처리 후 데이터 info ===")
print(df_clean.info())
print("\n=== 전처리 후 데이터 describe ===")
print(df_clean.describe())

#%% 2. 결측치 분포 비교 시각화 (범례 없이 색상 구분)
plt.figure(figsize=(10,6))
missing = pd.DataFrame({
    "Before": df_raw.isnull().sum(),
    "After": df_clean.isnull().sum()
}).fillna(0).astype(int).sort_values("Before", ascending=False)

missing.plot(kind='barh', color=['#E19992', '#D54344'], legend=False)
plt.title('결측치 비교: 전처리 전 vs 후', fontsize=14, fontweight='bold')
plt.ylabel('컬럼명')
plt.xlabel('결측치 개수')
plt.tight_layout()
plt.savefig('2.visualization/missing_values_before.png', dpi=300)
plt.show()

#%% 3. 라벨 매핑: 0,1 숫자를 의미 있는 이름으로 변환

depression_map = {0: 'No Depression', 1: 'Depression'}
gender_map = {1: 'Male', 2: 'Female'}

df_clean['Depression_label'] = df_clean['Depression'].map(depression_map)
df_clean['Gender_label'] = df_clean['Gender'].map(gender_map)

#%% 4. 주요 변수별 그룹 분포 시각화

# 4-1. 성별에 따른 우울증 분포
plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x='Gender_label', hue='Depression_label', palette=['#3d4e62', '#f8dcde'])
plt.title('성별에 따른 우울증 분포', fontsize=14)
plt.tight_layout()
plt.savefig('2.visualization/gender_vs_depression.png', dpi=300)
plt.show()

# 4-2. 우울증 여부에 따른 수면 시간 분포
plt.figure(figsize=(6,4))
sns.boxplot(data=df_clean, x='Depression_label', y='Sleep Duration', palette=['#3d4e62', '#f8dcde'])
plt.title('우울증 여부에 따른 수면 시간 분포', fontsize=14)
plt.tight_layout()
plt.savefig('2.visualization/sleep_vs_depression.png', dpi=300)
plt.show()

# 4-3. 우울증 여부에 따른 재정 스트레스 분포
plt.figure(figsize=(6,4))
sns.boxplot(data=df_clean, x='Depression_label', y='Financial Stress', palette=['#3d4e62', '#f8dcde'])
plt.title('우울증 여부에 따른 재정 스트레스 분포', fontsize=14)
plt.tight_layout()
plt.savefig('2.visualization/finance_vs_depression.png', dpi=300)
plt.show()



#%% 5. 수면시간 이상치 탐지 및 처리

# IQR 기반 이상치 기준 계산
Q1 = df_clean['Sleep Duration'].quantile(0.25)
Q3 = df_clean['Sleep Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"IQR 기준 하한: {lower_bound}, 상한: {upper_bound}")

# 이상치 마스킹
outlier_mask = (df_clean['Sleep Duration'] < lower_bound) | (df_clean['Sleep Duration'] > upper_bound)
df_outliers = df_clean[outlier_mask]
df_no_outliers = df_clean[~outlier_mask]

print(f"이상치 수: {df_outliers.shape[0]}")

# 이상치 제거 후 시각화
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_no_outliers, x='Depression_label', y='Sleep Duration', palette=['#3d4e62', '#f8dcde'])
plt.title('수면시간 이상치 제거 후 분포', fontsize=14)
plt.tight_layout()
plt.savefig('2.visualization/sleep_duration_no_outlier.png', dpi=300)
plt.show()

# 중앙값 대체 처리
median_val = df_no_outliers['Sleep Duration'].median()
df_median_filled = df_clean.copy()
df_median_filled.loc[outlier_mask, 'Sleep Duration'] = median_val

# 중앙값 대체 후 시각화
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_median_filled, x='Depression_label', y='Sleep Duration', palette=['#3d4e62', '#f8dcde'])
plt.title('수면시간 중앙값 대체 후 분포', fontsize=14)
plt.tight_layout()
plt.savefig('2.visualization/sleep_duration_filled_median.png', dpi=300)
plt.show()
