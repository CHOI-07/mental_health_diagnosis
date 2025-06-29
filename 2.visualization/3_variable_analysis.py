#%% 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform
from pathlib import Path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# 폰트 설정 (mac + windows 대응)
if platform.system() == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rcParams["axes.unicode_minus"] = False

#%% 경로 설정 (macOS, Windows 모두 호환)
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "0.preprocessing" / "mental_train_preprocessed.csv"
save_dir = base_dir / "2.visualization"
save_dir.mkdir(parents=True, exist_ok=True)

#%% 데이터 로드 및 통합 변수 생성
df = pd.read_csv(data_path)
df['Satisfaction'] = df[['Study Satisfaction', 'Job Satisfaction']].mean(axis=1)
df['Pressure'] = df[['Academic Pressure', 'Work Pressure']].mean(axis=1)

# Depression 컬럼을 텍스트 라벨로 변환
df['Depression_Label'] = df['Depression'].map({0: 'No Depression', 1: 'Depression'})
target = 'Depression_Label'  # target 변수를 라벨 버전으로 변경

#%% 변수 및 시각화 스타일 설정 (target 변수 정의를 이리로 이동)
core_cols = ['Age', 'Sleep Duration', 'Satisfaction', 'Pressure', 'Financial Stress', 'Work/Study Hours']
colors = ['#3d4e62', '#8a8a8a']  # 어두운 파란색게열, 밝은 회색

#%% 범주형 변수 리스트 생성
# 수치형이 아닌 모든 컬럼을 범주형으로 간주 (Depression 제외, id/Name 등 제외)
exclude_cols = ['id', 'Name']
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [col for col in df.columns if col not in numeric_cols + [target] + exclude_cols]

#%% 1. 전체 상관관계 히트맵
# 네이비-실버-스카이블루 그라데이션 컬러맵 생성
custom_cmap_corr = LinearSegmentedColormap.from_list(
    "navy_silver_skyblue",
    ["#223A5E", "#e0e5ec", "#6FB1FC"]
)

import numpy as np
corr = df.corr(numeric_only=True)
mask = np.abs(corr.values) < 0.3
annot = corr.round(2).astype(str).values
annot[mask] = ""

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap=custom_cmap_corr, center=0, annot=annot, fmt='', cbar=True)
plt.title("전체 변수 간 상관관계 히트맵", fontsize=14)
plt.xticks(rotation=45)  # x축 라벨 45도 회전
plt.tight_layout()
plt.savefig(save_dir / "3_1_variable_all_correlation_heatmap.png")
plt.close()

#%% 2. 핵심 변수 Spearman 상관관계
plt.figure(figsize=(8, 6))
sns.heatmap(df[core_cols].corr(method='spearman'), cmap='vlag', annot=True, vmin=-1, vmax=1)
plt.title("Spearman 상관관계 히트맵", fontsize=13)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_dir / "3_2_variable_spearman_corr_heatmap.png")
plt.close()

#%% 3. Depression 그룹별 핵심 변수 평균 비교
# 그룹별 평균 계산
mean_df = df.groupby(target)[core_cols].mean().T
# 변수별 평균 차이의 절대값이 큰 순서로 정렬
mean_df['diff'] = abs(mean_df.iloc[:,0] - mean_df.iloc[:,1])
mean_df = mean_df.sort_values('diff', ascending=False).drop('diff', axis=1)
ax = mean_df.plot(kind='bar', figsize=(10, 6), color=colors)
plt.title("Depression 그룹별 핵심 변수 평균 비교", fontsize=13)
plt.ylabel("평균값")
plt.xticks(rotation=45)
# 각 막대 위에 평균값 표시
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=9, rotation=0)
plt.tight_layout()
plt.savefig(save_dir / "3_3_mean_numeric_by_depression.png")
plt.close()

#%% 4. 타겟 분포 파이차트
labels = ['No Depression', 'Depression']
sizes = df[target].value_counts(normalize=True).sort_index()
counts = df[target].value_counts().sort_index()
explode = [0.08, 0]  # No Depression만 살짝 분리
colors = ['#efb0b0', '#c24444']  # 연핑크, 진레드

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(counts))}명)',
    colors=colors,
    explode=explode,
    startangle=140,
    textprops={'fontsize': 13, 'weight': 'bold', 'color': 'black'},
    wedgeprops={'width': 0.4}  # 도넛형
)
plt.setp(autotexts, size=13, weight='bold', color='black')
plt.setp(texts, size=13, color='black')
ax.set_title("Depression 타겟 분포", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(save_dir / "3_4_target_distribution_pie.png")
plt.close()

#%% 5. 범주형 변수 비율 시각화 (예: Gender)
# 유의미한 범주형 변수 자동 선별 + 시각화
threshold = 0.15  # 두 그룹 간 비율 차이가 15% 이상이면 유의미하다고 판단
selected_cats = []

for col in categorical_cols:
    # 샘플 수가 5개 미만인 범주는 제외
    if df[col].value_counts().min() < 5:
        continue
    prop_series = df.groupby([col, target]).size().groupby(level=0).apply(lambda x: x / x.sum())
    prop_series.name = 'ratio'
    # 인덱스가 MultiIndex인지 확인
    if isinstance(prop_series.index, pd.MultiIndex):
        names = list(prop_series.index.names)
        if names[0] == 'Name':
            names[0] = 'Category'
        prop_series.index = prop_series.index.set_names(names)
        idx_name = names[0]
    else:
        idx_name = col if col != 'Name' else 'Category'
        prop_series.index = prop_series.index.set_names(idx_name)
    prop_df = prop_series.reset_index()
    pivot = prop_df.pivot(index=idx_name, columns=target, values='ratio')
    if pivot.shape[1] == 2:
        max_diff = (pivot.max(axis=1) - pivot.min(axis=1)).max()
        if max_diff >= threshold:
            selected_cats.append(col)

# 유의미한 변수만 시각화
for col in selected_cats:
    plt.figure(figsize=(6, 4))
    plot_series = (
        df.groupby([col, target]).size()
        .groupby(level=0).apply(lambda x: x / x.sum())
    )
    plot_series.name = 'ratio'
    if isinstance(plot_series.index, pd.MultiIndex):
        names = list(plot_series.index.names)
        if names[0] == 'Name':
            names[0] = 'Category'
        plot_series.index = plot_series.index.set_names(names)
        idx_name = names[0]
    else:
        idx_name = col if col != 'Name' else 'Category'
        plot_series.index = plot_series.index.set_names(idx_name)
    plot_df = plot_series.reset_index()
    # 컬럼 개수 자동 맞추기
    if len(plot_df.columns) > 3:
        plot_df = plot_df.iloc[:, -3:]
    plot_df.columns = [idx_name, target, 'ratio']
    sns.barplot(x=idx_name, y='ratio', hue=target, data=plot_df, palette=colors)
    plt.title(f"{col}별 {target} 비율", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / f"3_5_ratio_{col}_vs_{target}.png")
    plt.close()

#%% 6. 주요 수치형 변수의 박스플롯 (Depression 그룹별)
for i, col in enumerate(core_cols, 1):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=col, data=df, palette=colors)
    plt.title(f"{col} 분포 (Depression 그룹별)")
    plt.tight_layout()
    plt.savefig(save_dir / f"3_7_boxplot_{i}_{col}_by_{target}.png")
    plt.close()