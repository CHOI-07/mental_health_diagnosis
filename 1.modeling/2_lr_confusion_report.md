# 2. Logistic Regression 이진 분류 모델 실험 로그

---

## What: 작업 목적

- 전처리된 정신건강 자가진단 데이터를 기반으로 우울증 여부 분류
- 학생 / 직장인 그룹을 분리하여 각각 Logistic Regression 모델 학습
- 모델: `LogisticRegression(max_iter=1000)`

---

## How: 실험 방식

- 입력 데이터: `mental_train_preprocessed.csv`
- 타겟 변수: `Depression`
- 그룹 분리 기준: `Working/Student` 컬럼 기반
- 학습 방식:
  - 그룹별 train/test 분리
  - 범주형 변수: `get_dummies` 인코딩
  - train/test 컬럼 정렬 통일 (`reindex`)
- 모델 평가:
  - classification_report → `.txt` 저장
  - confusion matrix → `.png` 저장 (텍스트 자동 색상 조절)

---

## Why: 실험 의도

- 비교적 해석 용이한 선형 모델(Logistic Regression) 기반의 baseline 성능 확보
- 학생/직장인 그룹의 클래스 불균형 대응력 및 recall 분석
- 후속 트리 기반 모델(RandomForest, XGBoost 등)과의 성능 비교 목적
- confusion matrix 이미지화는 발표용 시각 자료 및 포트폴리오 삽입용 활용

---

## 평가 요약

### ▶ 학생 그룹 (Students)
- 정확도: **0.83**
- 주요 지표:
  - F1-score (우울 있음): **0.85**
  - Recall (우울 있음): **0.87**
  - Recall (비우울): **0.77**

### ▶ 직장인 그룹 (Professionals)
- 정확도: **0.96**
- 주요 지표:
  - F1-score (우울 있음): **0.71**
  - Recall (우울 있음): **0.64**
  - Precision (우울 있음): **0.80**
  - 불균형 영향 → 민감도(Recall) 저하 우려

---

## 아웃풋 파일

- `logreg_students_report.txt`, `logreg_professionals_report.txt`
- `cm_logreg_students.png`, `cm_logreg_professionals.png`
