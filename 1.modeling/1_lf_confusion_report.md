# 1_lf_confusion_report

## RandomForest 모델 평가 결과

### 1. 학생 (Students) 집단

#### Confusion Matrix
![Confusion Matrix - Students](./1.modeling/cm_students.png)

| 실제\예측       | No Depression (0) | Depression (1) |
|-----------------|-------------------|----------------|
| No Depression (0) | 1815              | 528            |
| Depression (1)    | 405               | 2833           |

- Accuracy: 0.83
- Precision, Recall, F1-score (상세한 표는 코드 내 classification_report 참고)

---

### 2. 직장인 (Professionals) 집단

#### Confusion Matrix
![Confusion Matrix - Professionals](./1.modeling/cm_professionals.png)

| 실제\예측       | No Depression (0) | Depression (1) |
|-----------------|-------------------|----------------|
| No Depression (0) | 20000             | 189            |
| Depression (1)    | 749               | 1141           |

- Accuracy: 0.96
- Precision, Recall, F1-score (상세한 표는 코드 내 classification_report 참고)

---

## 분석 요약

- 직장인 모델은 No Depression 예측에서 매우 높은 정확도를 보이나 Depression 예측에서는 상대적으로 낮은 recall(0.60)을 기록해 개선 여지가 있음.
- 학생 모델은 비교적 균형 잡힌 성능을 보이며, 추후 feature 선택 및 하이퍼파라미터 튜닝으로 성능 개선 가능.
- confusion matrix는 모델 오분류 패턴을 파악하는 데 유용하며, 실제 서비스 적용 전 오탐/미탐 위험 평가에 반드시 활용 필요.

---

> **참고**  
> - 클래스 라벨 0: No Depression  
> - 클래스 라벨 1: Depression  
> - 평가 지표는 `classification_report` 함수 결과를 기준으로 작성되었습니다.
