---
title: 3_shap_summary_professionals
created: 2025-06-26
updated: 2025-06-26
tags: [SHAP, 직장인, XGBoost, 중요변수]
---

## 🎯 직장인 그룹 SHAP 분석 결과 요약

### 분석 대상
- 모델: XGBoost 재학습 모델 (`xgb_professionals_model.pkl`)
- 대상: `Working Professional or Student == 2`에 해당하는 직장인 데이터
- SHAP 기반 모델 해석 수행

---

### 📌 주요 Feature 중요도 (bar plot 기준)

- **Age**: +2.14  
- **Have you ever had suicidal thoughts?**: +1.00  
- **Job Satisfaction**, **Financial Stress**, **Work Pressure** 등 직무 관련 지표가 상위에 분포  

해당 변수들은 우울감 예측에 있어 큰 영향을 미치는 것으로 해석됨

이미지 저장:
- `shap_bar_professionals.png`
- `shap_summary_professionals.png`

---

### 🔍 SHAP Summary 해석

- `Age` 및 `자살 생각 경험` 여부가 전체 예측에서 핵심적인 영향
- `Job Satisfaction`, `Financial Stress`는 상대적으로 feature 값이 클수록 예측값에 + 방향으로 작용
- 파란색(낮은 값) → 우울감 예측에 부정적 기여  
  빨간색(높은 값) → 우울감 예측에 긍정적 기여

---

### ⏭ 다음 작업
- 학생 그룹의 SHAP 해석과 비교하여 시각적 대조 분석 수행  
- 상위 5~7개 feature를 중심으로 직관적인 설명 그래프 포트폴리오용으로 정제  
