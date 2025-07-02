#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 폰트 설정 (mac + windows 대응)
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 불러오기
path = 'mental_train.csv'
df = pd.read_csv(path, na_values=['', np.nan])

# print("현재 컬럼 목록:", df.columns.tolist())

# # 수면시간 컬럼 숫자형 변환 (문자열 포함된 경우 NaN 처리)
# df['Sleep Duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')

# #%% 이상치 제거 (Sleep Duration - IQR 방식)
# Q1 = df['Sleep Duration'].quantile(0.25)
# Q3 = df['Sleep Duration'].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR
# print(f"수면시간 IQR 기준: {lower:.2f} ~ {upper:.2f}")
# print(f"이상치 제거 수: {(df['Sleep Duration'] < lower).sum() + (df['Sleep Duration'] > upper).sum()}개")
# df = df[(df['Sleep Duration'] >= lower) & (df['Sleep Duration'] <= upper)]

# #%% 문자열 매핑
# mixed_mapping = {'Student': 1, 'Working Professional': 2, 'Intern': 3}
# df['Working Professional or Student'] = df['Working Professional or Student'].map(mixed_mapping)
# suicidal_mapping = {'No': 1, 'Yes': 2}
# df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(suicidal_mapping)
# gender_mapping = {'Male': 1, 'Female': 2}
# df['Gender'] = df['Gender'].map(gender_mapping)
# hist_mapping = {'No': 1, 'Yes': 2}
# df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(hist_mapping)
# degree_mapping = {'Bachelor': 2, 'Master': 3, 'PhD': 4}
# df['Degree'] = df['Degree'].map(degree_mapping)

# #%% factorize
# cols_to_factorize = ['Dietary Habits', 'City', 'Profession']
# for col in cols_to_factorize:
#     df[col], _ = pd.factorize(df[col])

# #%% 결측치 처리
# df['Profession'] = df['Profession'].fillna(0).astype(int) + 1
# df['Degree'] = df['Degree'].fillna(0).astype(int) + 1
# empty_columns = ['Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work Pressure', 'Job Satisfaction']
# df[empty_columns] = df[empty_columns].fillna(0)
# df[['Financial Stress']] = df[['Financial Stress']].fillna(df[['Financial Stress']].mean())

# #%% 파생 변수 생성
# def compute_total_pressure(row):
#     academic = row['Academic Pressure']
#     work = row['Work Pressure']
#     if pd.notna(academic) and pd.notna(work):
#         return (academic + work) / 2
#     elif pd.notna(academic):
#         return academic
#     elif pd.notna(work):
#         return work
#     else:
#         return np.nan
# def compute_total_satisfaction(row):
#     study = row['Study Satisfaction']
#     job = row['Job Satisfaction']
#     if pd.notna(study) and pd.notna(job):
#         return (study + job) / 2
#     elif pd.notna(study):
#         return study
#     elif pd.notna(job):
#         return job
#     else:
#         return np.nan
# df['Total Pressure'] = df.apply(compute_total_pressure, axis=1)
# df['Total Satisfaction'] = df.apply(compute_total_satisfaction, axis=1)

#%% 전처리 검증
print("[describe()] 요약:")
print(df.describe(include='all').T)

print("\n[dtypes] 요약:")
print(df.dtypes.value_counts())

print("\n[결측치] 상위 20개:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

total_missing = df.isnull().sum().sum()
print(f"\n[전체 결측치 수]: {total_missing:,}개")

#%% 시각화용 비교용 데이터 생성
df_raw = pd.read_csv("mental_train.csv", na_values=["", np.nan])
df_clean = pd.read_csv("0.preprocessing/mental_train_preprocessed.csv")

custom_palette = {
    "Before": "#E19992",
    "After": "#D54344"
}

missing = pd.DataFrame({
    "Before": df_raw.isnull().sum(),
    "After": df_clean.isnull().sum()
}).fillna(0).astype(int).sort_values("Before", ascending=False)

types_raw = df_raw.dtypes.value_counts()
types_clean = df_clean.dtypes.value_counts()
dtype_df = pd.DataFrame({
    "Before": types_raw,
    "After": types_clean
}).fillna(0).astype(int)



# 상위 7개 결측치만 추출
top_missing = missing[missing["Before"] > 0].head(7)

x = np.arange(len(top_missing.index))  # y축 위치
bar_width = 0.35

fig3, ax3 = plt.subplots(figsize=(10, 6))
bars_before = ax3.barh(x - bar_width/2, top_missing["Before"], height=bar_width, color=custom_palette["Before"], label="Before")
bars_after = ax3.barh(x + bar_width/2, top_missing["After"], height=bar_width, color=custom_palette["After"], label="After")

# 수치 라벨 추가 (Before)
for bar in bars_before:
    width = bar.get_width()
    if width > 0:
        ax3.text(width + 3000, bar.get_y() + bar.get_height()/2,
                f"{int(width):,}", va='center', fontsize=10, color='black')
# 수치 라벨 추가 (After)
for bar in bars_after:
    width = bar.get_width()
    if width > 0:
        ax3.text(width + 3000, bar.get_y() + bar.get_height()/2,
                f"{int(width):,}", va='center', fontsize=10, color='black')

ax3.set_yticks(x)
ax3.set_yticklabels(top_missing.index)
ax3.set_title("결측치 비교: 전처리 전 vs 후 (상위 7개)", fontsize=14, fontweight='bold')
ax3.set_xlabel("결측치 수")
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("missing_comparison_top7.png", dpi=300)
plt.close()
