# 3. XGBoost 이진 분류 모델 실험 로그

---

## What: 작업 목적

- 전처리된 정신건강 자가진단 데이터를 기반으로 XGBoostClassifier를 활용한 우울증 분류
- 직업군(학생/직장인) 분리 후 개별 모델 학습
- 트리 기반 Boosting 기법을 통해 Recall 성능 및 예측 정밀도 개선 시도

---

## How: 실험 방식

- 입력 데이터: `mental_train_preprocessed.csv`
- 타겟 변수: `Depression`
- 전처리
  - 파생 변수(`Total Pressure`, `Total Satisfaction`)만 사용
  - One-hot 인코딩 후 train/test 컬럼 정렬 맞춤
- 모델 학습
  - `XGBClassifier(n_estimators=100, use_label_encoder=False)`
  - 학생/직장인 별도 학습
- 평가
  - classification_report `.txt` 저장
  - confusion matrix `.png` 저장

---

## Why: 실험 의도

- 기존 RandomForest 대비 성능 향상 여부 확인
- 우울 클래스(소수 클래스) 예측 Recall 개선이 주요 목표
- Boosting 계열 모델의 일반화 성능 확인 및 해석 자료 확보 목적

---

## 평가 요약

### ▶ 학생 그룹 (Students)
- 정확도: **0.84**
- 주요 지표:
  - F1-score (우울 있음): **0.86**
  - Recall (우울 있음): **0.87**
  - Precision (우울 있음): **0.85**

### ▶ 직장인 그룹 (Professionals)
- 정확도: **0.96**
- 주요 지표:
  - F1-score (우울 있음): **0.75**
  - Recall (우울 있음): **0.70**
  - Precision (우울 있음): **0.81**

---

## 아웃풋 파일

- `xgb_students_report.txt`, `xgb_professionals_report.txt`
- `cm_xgb_students.png`, `cm_xgb_professionals.png`
