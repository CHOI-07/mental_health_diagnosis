# 3_xgb_confusion_report

## XGBoost 분류 결과 및 혼동 행렬 시각화

### 1. 개요
- 정신건강 자가진단 데이터셋에서 학생/직장인 그룹별 분류
- XGBoost 분류기 사용, Accuracy, Precision, Recall, F1-score 평가
- 혼동 행렬 시각화 포함

### 2. 주요 평가 결과

| 그룹          | Accuracy | Precision (No/Yes) | Recall (No/Yes) | F1-score (No/Yes) |
|---------------|----------|--------------------|-----------------|-------------------|
| 학생 그룹     | 84%      | 0.82 / 0.85        | 0.79 / 0.87     | 0.80 / 0.86       |
| 직장인 그룹   | 96%      | 0.97 / 0.81        | 0.98 / 0.70     | 0.98 / 0.75       |

### 3. 혼동 행렬 이미지

- 학생 그룹 ![](./cm_xgb_students.png)  
- 직장인 그룹 ![](./cm_xgb_professionals.png)

---

### 4. 인사이트
- 학생 그룹은 균형 잡힌 성능  
- 직장인 그룹은 우울증 환자 재현율 낮음 → 데이터 보강 필요  

### 5. 파일
- 리포트: `1.modeling/xgb_students_report.txt`, `1.modeling/xgb_professionals_report.txt`  
- 이미지: `1.modeling/cm_xgb_students.png`, `1.modeling/cm_xgb_professionals.png`
