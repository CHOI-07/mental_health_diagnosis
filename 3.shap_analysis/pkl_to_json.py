import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 예시 데이터 (f1_df를 사용했다고 가정)
f1_df = pd.DataFrame({
    "Group": ["학생", "직장인"],
    "Precision": [0.925, 0.975],
    "Recall": [0.922, 0.973],
    "F1-score": [0.920, 0.972]
})
f1_melted = f1_df.melt(id_vars="Group", var_name="Metric", value_name="Score")

# 색상 팔레트 적용
palette = {"학생": "#3d4e62", "직장인": "#f8dcde"}

# 시각화
plt.figure(figsize=(8, 5))
sns.barplot(data=f1_melted, x="Metric", y="Score", hue="Group", palette=palette)
plt.title("학생 vs 직장인 모델 성능 비교")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("3.shap_analysis/slide13_model_performance_compare_korean.png", dpi=300)
plt.clf()
