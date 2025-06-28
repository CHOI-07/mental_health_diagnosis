# 1. RandomForest 이진 분류 모델 실험 로그

---

## What: 작업 목적

- 전처리된 정신건강 설문 데이터를 기반으로 우울증 여부를 예측
- 직업군(학생/직장인) 데이터 분리 후 각각 별도 모델 학습
- 모델: `RandomForestClassifier(n_estimators=100)`

---

## How: 실험 방식

- 입력 데이터: `mental_train_preprocessed.csv`
- 타겟 변수: `Depression`
- 제거 변수: 원본 스트레스/만족도 관련 항목 제거 (파생변수로 대체)
- 전처리
  - `get_dummies` 후 train/test 컬럼 정렬 강제 (reindex)
- 모델 학습
  - 학생용, 직장인용 랜덤포레스트 모델 각각 학습
- 평가
  - classification_report 저장
  - confusion matrix 이미지 저장 (글자 대비 자동조절)
  - feature importance `.csv` 저장

---

## Why: 실험 의도

- 학생/직장인 간 데이터 분포 및 변수 중요도 차이를 분리 모델로 분석
- 컬럼 정렬은 train/test 불일치로 인한 학습 오류 방지
- feature importance는 후속 변수 해석, EDA, 포트폴리오용으로 활용
- confusion matrix의 수동 텍스트 조정은 발표/시각화 목적 시 직관성 향상

---

## 아웃풋 파일

- `rf_students_report.txt`, `rf_professionals_report.txt`
- `cm_rf_students.png`, `cm_rf_professionals.png`
- `rf_students_feature_importance.csv`, `rf_professionals_feature_importance.csv`
