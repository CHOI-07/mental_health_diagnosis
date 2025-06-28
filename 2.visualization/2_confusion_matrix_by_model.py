# #%% [1] 라이브러리 로드
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from pathlib import Path

# # 포폴용 색상 팔레트
# custom_colors = ["#3d4e62", "#E19992", "#D54344"]

# #%% [2] 경로 설정 (OS에 따라 자동 처리)
# base_dir_mac = Path.home() / "Desktop/ME/1.re_project/정신건강_자가진단"
# base_dir_win = Path("C:/Users/hhhey/Desktop/ME/1.re_project/정신건강_자가진단")

# base_dir = base_dir_mac if base_dir_mac.exists() else base_dir_win
# data_path = base_dir / "0.preprocessing" / "mental_train_preprocessed.csv"
# model_path_students = base_dir / "3.shap_analysis" / "xgb_students_model.pkl"
# model_path_workers = base_dir / "3.shap_analysis" / "xgb_professionals_model.pkl"
# feature_path = base_dir / "3.shap_analysis" / "feature_columns.txt"
# save_dir = base_dir / "2.visualization"  # ← 여기서 4 ➝ 2 로 변경
# save_dir.mkdir(parents=True, exist_ok=True)

# #%% [3] 데이터 로드
# df = pd.read_csv(data_path)
# X = df.drop(columns=["Depression", "Name"])
# y = df["Depression"]
# group = df["Working Professional or Student"]

# # 전체 get_dummies
# X_encoded = pd.get_dummies(X, drop_first=True)
# with open(feature_path, "r", encoding="utf-8") as f:
#     feature_order = f.read().splitlines()
# X_encoded = X_encoded[feature_order]

# X_students = X_encoded[group == 1]
# y_students = y[group == 1]
# X_workers = X_encoded[group == 2]
# y_workers = y[group == 2]

# #%% [4] 모델 로드
# model_students = joblib.load(model_path_students)
# model_workers = joblib.load(model_path_workers)

# #%% [5] 예측 및 혼동행렬 시각화 함수
# def plot_confusion(y_true, y_pred, group_name, save_path):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
#     fig, ax = plt.subplots(figsize=(5, 4))
#     disp.plot(ax=ax, cmap=sns.color_palette(custom_colors, as_cmap=True))
#     ax.set_title(f"{group_name} 그룹 혼동행렬", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


# #%% [6] 혼동행렬 저장
# plot_confusion(y_students, model_students.predict(X_students), "학생", save_dir / "confusion_students.png")
# plot_confusion(y_workers, model_workers.predict(X_workers), "직장인", save_dir / "confusion_workers.png")
# print("혼동행렬 시각화 저장 완료")



#%% [1] 라이브러리 로드
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib
import platform
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import colormaps

# 폰트 설정 (mac + windows 대응)
if platform.system() == "Darwin":
    matplotlib.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rcParams["axes.unicode_minus"] = False

# 포폴용 컬러
main_color = "#3d4e62"
sub_color = "#556677"
heatmap_cmap = sns.light_palette(main_color, as_cmap=True)

#%% [2] 경로 설정 (OS별 자동 전환)
base_dir_mac = Path.home() / "Desktop/ME/1.re_project/정신건강_자가진단"
base_dir_win = Path("C:/Users/hhhey/Desktop/ME/1.re_project/정신건강_자가진단")
base_dir = base_dir_mac if base_dir_mac.exists() else base_dir_win

data_path = base_dir / "0.preprocessing" / "mental_train_preprocessed.csv"
model_path_students = base_dir / "3.shap_analysis" / "xgb_students_model.pkl"
model_path_workers = base_dir / "3.shap_analysis" / "xgb_professionals_model.pkl"
feature_path = base_dir / "3.shap_analysis" / "feature_columns.txt"
save_dir = base_dir / "2.visualization"
save_dir.mkdir(parents=True, exist_ok=True)

#%% [3] 데이터 및 모델 로드
df = pd.read_csv(data_path)
X = df.drop(columns=["Depression", "Name"])
y = df["Depression"]
group = df["Working Professional or Student"]

X_encoded = pd.get_dummies(X, drop_first=True)
with open(feature_path, "r", encoding="utf-8") as f:
    feature_order = f.read().splitlines()
X_encoded = X_encoded[feature_order]

X_students = X_encoded[group == 1]
y_students = y[group == 1]
X_workers = X_encoded[group == 2]
y_workers = y[group == 2]

model_students = joblib.load(model_path_students)
model_workers = joblib.load(model_path_workers)

#%% [4] 혼동행렬 시각화 함수
def plot_confusion(y_true, y_pred, group_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    # 색상 팔레트
    cmap = sns.light_palette("#3d4e62", as_cmap=True)
    norm = Normalize(vmin=cm.min(), vmax=cm.max())
    sns.heatmap(cm, annot=False, fmt="d", cmap=cmap, cbar=False,
                linewidths=0.5, linecolor="#a0a0a0", ax=ax)

    # 텍스트 색상 자동 설정
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            r, g, b, _ = cmap(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:,}",  # 천 단위 쉼표
                    ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

    # 제목 및 축 라벨
    ax.set_title(f"{group_name} 그룹 혼동행렬", fontsize=16, color="#3d4e62", pad=20)
    ax.set_xlabel("Predicted label", fontsize=12, color="#556677")
    ax.set_ylabel("True label", fontsize=12, color="#556677")
    ax.set_xticklabels(["No", "Yes"], fontsize=11, color="#3d4e62")
    ax.set_yticklabels(["No", "Yes"], fontsize=11, color="#3d4e62", rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#%% [5] 시각화 저장
plot_confusion(y_students, model_students.predict(X_students), "학생", save_dir / "confusion_students.png")
plot_confusion(y_workers, model_workers.predict(X_workers), "직장인", save_dir / "confusion_workers.png")
print("혼동행렬 시각화 저장 완료")
