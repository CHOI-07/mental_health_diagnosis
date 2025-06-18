#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic 경로
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False


# 데이터 불러오기 (경로 수정 필요)
path = 'mental_train.csv'
df = pd.read_csv(path, na_values=['', np.nan])


print("✅ 현재 컬럼 목록:", df.columns.tolist())


#%% 문자열 매핑 + 라벨 컬럼 병행 생성
mixed_mapping = {'Student': 1, 'Working Professional': 2, 'Intern': 3}
df['Working Professional or Student'] = df['Working Professional or Student'].map(mixed_mapping)

suicidal_mapping = {'No':1,'Yes':2}
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(suicidal_mapping)

gender_mapping = {'Male': 1, 'Female': 2}
df['Gender'] = df['Gender'].map(gender_mapping)

hist_mapping = {'No':1,'Yes':2}
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(hist_mapping)

degree_mapping = {'Bachelor': 2, 'Master': 3, 'PhD': 4}
df['Degree'] = df['Degree'].map(degree_mapping)

#%% factorize
cols_to_factorize = ['Sleep Duration', 'Dietary Habits', 'City', 'Profession']
for col in cols_to_factorize:
    df[col], _ = pd.factorize(df[col])

#%% 결측치 처리
df['Profession'] = df['Profession'].fillna(0).astype(int) + 1
df['Degree'] = df['Degree'].fillna(0).astype(int) + 1

empty_columns = ['Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work Pressure', 'Job Satisfaction']
df[empty_columns] = df[empty_columns].fillna(0)

df[['Financial Stress']] = df[['Financial Stress']].fillna(df[['Financial Stress']].mean())

#%% 파생 변수 생성

def compute_total_pressure(row):
    academic = row['Academic Pressure']
    work = row['Work Pressure']
    if pd.notna(academic) and pd.notna(work):
        return (academic + work) / 2
    elif pd.notna(academic):
        return academic
    elif pd.notna(work):
        return work
    else:
        return np.nan

def compute_total_satisfaction(row):
    study = row['Study Satisfaction']
    job = row['Job Satisfaction']
    if pd.notna(study) and pd.notna(job):
        return (study + job) / 2
    elif pd.notna(study):
        return study
    elif pd.notna(job):
        return job
    else:
        return np.nan

# 적용
df['Total Pressure'] = df.apply(compute_total_pressure, axis=1)
df['Total Satisfaction'] = df.apply(compute_total_satisfaction, axis=1)


#%% 전처리 검증: describe, info, 결측치 확인

# 기본 통계 확인
print("✅ [describe()] 요약:")
print(df.describe(include='all').T)

# 데이터 타입 요약
print("\n✅ [dtypes] 요약:")
print(df.dtypes.value_counts())

# 결측치 확인
print("\n✅ [결측치] 상위 20개:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 전체 결측치 개수
total_missing = df.isnull().sum().sum()
print(f"\n✅ [전체 결측치 수]: {total_missing:,}개")

#%% 전처리된 파일 저장
df.to_csv("mental_train_preprocessed.csv", index=False)


#%% 4. 전처리 전후 비교용 시각화
# 원본 다시 불러와 전처리 전 상태 비교



df_raw = pd.read_csv("mental_train.csv", na_values=["", np.nan])
df_clean = pd.read_csv("mental_train_preprocessed.csv")

custom_palette = {
    "Before": "#E19992",
    "After": "#D54344"
}

# 결측치 비교
missing = pd.DataFrame({
    "Before": df_raw.isnull().sum(),
    "After": df_clean.isnull().sum()
}).fillna(0).astype(int).sort_values("Before", ascending=False)

# 데이터 타입 비교
types_raw = df_raw.dtypes.value_counts()
types_clean = df_clean.dtypes.value_counts()
dtype_df = pd.DataFrame({
    "Before": types_raw,
    "After": types_clean
}).fillna(0).astype(int)

#%% 5. 결측치 그래프 저장
fig1, ax1 = plt.subplots(figsize=(10, 6))
missing.plot(kind="barh", ax=ax1,
             color=[custom_palette["Before"], custom_palette["After"]])
ax1.set_title("결측치 비교: 전처리 전 vs 후", fontsize=14, fontweight='bold', color='#000000')
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("missing_comparison.png", dpi=300, facecolor='white')

#%% 6. 데이터 타입 그래프 저장
fig2, ax2 = plt.subplots(figsize=(8, 6))
dtype_df.plot(kind="bar", ax=ax2,
              color=[custom_palette["Before"], custom_palette["After"]],
              legend=False)
ax2.set_title("데이터 타입 개수 변화", fontsize=14, fontweight='bold', color='#000000')
ax2.tick_params(axis='x', labelrotation=0, labelsize=11)
ax2.legend(["Before", "After"],
           loc='lower center',
           bbox_to_anchor=(0.5, -0.25),
           ncol=2,
           frameon=False,
           fontsize=10)
plt.tight_layout()
plt.savefig("dtype_comparison.png", dpi=300, facecolor='white')

# %%
