#%%
# 분석 목적 및 데이터 불러오기
#%%
import numpy as np
import pandas as pd

# 데이터 불러오기 (경로 수정 필요)
path = 'mental_train_lot,lat_최종.csv'
df = pd.read_csv(path, na_values=['', np.nan])

#%%
# 전처리: 문자열을 숫자형으로 매핑 + 라벨 컬럼 병행 생성
#%%
mixed_mapping = {'Student': 1, 'Working Professional': 2, 'Intern': 3}
mixed_reverse = {v: k for k, v in mixed_mapping.items()}
df['Working Professional or Student'] = df['Working Professional or Student'].map(mixed_mapping)
df['Job_label'] = df['Working Professional or Student'].map(mixed_reverse)

suicidal_mapping = {'No': 1, 'Yes': 2}
suicidal_reverse = {v: k for k, v in suicidal_mapping.items()}
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(suicidal_mapping)
df['SuicideThought_label'] = df['Have you ever had suicidal thoughts ?'].map(suicidal_reverse)

gender_mapping = {'Male': 1, 'Female': 2}
gender_reverse = {v: k for k, v in gender_mapping.items()}
df['Gender'] = df['Gender'].map(gender_mapping)
df['Gender_label'] = df['Gender'].map(gender_reverse)

hist_mapping = {'No': 1, 'Yes': 2}
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(hist_mapping)

sleep_mapping = {'Less than 5 hours': 1, '5-6 hours': 2, '6-7 hours': 3, '7-8 hours': 4, 'exceeding 8 hours': 5}
sleep_reverse = {v: k for k, v in sleep_mapping.items()}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)
df['Sleep_label'] = df['Sleep Duration'].map(sleep_reverse)

#%%
# factorize된 라벨 보존
#%%
df['Dietary Habits'], dietary_labels = pd.factorize(df['Dietary Habits'])
df['Dietary_label'] = df['Dietary Habits'].map(lambda x: dietary_labels[x])

df['City'], city_labels = pd.factorize(df['City'])
df['City_label'] = df['City'].map(lambda x: city_labels[x])

df['Profession'], prof_labels = pd.factorize(df['Profession'])
df['Profession'] = df['Profession'].fillna(0).astype(int) + 1
df['Profession_label'] = df['Profession'].map(lambda x: prof_labels[x - 1] if x > 0 else 'Unknown')

#%%
# 결측치 처리
#%%
df[['Pressure', 'Satisfaction']] = df[['Pressure', 'Satisfaction']].fillna(0)
df[['Financial Stress']] = df[['Financial Stress']].fillna(df[['Financial Stress']].mean())
df[['lat']] = df[['lat']].fillna(df[['lat']].mean())
df[['log']] = df[['log']].fillna(df[['log']].mean())

#%%
# 모델 학습 (XGBoost)
#%%
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 불필요한 열 제거
if 'id' in df.columns and 'Name' in df.columns:
    df = df.drop(columns=['id', 'Name'])

target_variable = 'Depression'
X = df.drop(columns=[target_variable])
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

#%%
# 시각화: confusion matrix, classification report, 주요 barplot
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (XGBoost)")
plt.show()

# classification report heatmap
report_dict = classification_report(y_test, y_pred_xgb, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :].T
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report (XGBoost)")
plt.tight_layout()
plt.show()

# barplot: 주요 변수 vs 우울증
sns.barplot(x='Gender_label', y='Depression', data=df, ci=None)
plt.title("Gender vs Depression")
plt.show()

sns.barplot(x='Dietary_label', y='Depression', data=df, ci=None)
plt.xticks(rotation=45)
plt.title("Dietary Habits vs Depression")
plt.show()

sns.barplot(x='Sleep_label', y='Depression', data=df, ci=None)
plt.title("Sleep Duration vs Depression")
plt.show()

#%%
# 지도 시각화 (Folium)
#%%
import folium

latitude_mean = df['lat'].mean()
longitude_mean = df['log'].mean()
map_center = [latitude_mean, longitude_mean]

map = folium.Map(location=map_center, zoom_start=5)

for i in df.index:
    depression_status = df.loc[i, 'Depression']
    color = 'red' if depression_status == 2 else 'green'

    folium.Circle(
        location=[df.loc[i, 'lat'], df.loc[i, 'log']],
        tooltip=f"City: {df.loc[i, 'City_label']}, Depression: {depression_status}",
        radius=100,
        color=color,
        fill=True,
        fill_color=color
    ).add_to(map)

map.save("mini_map.html")
