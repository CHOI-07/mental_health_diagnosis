---
title: 02_혼동행렬_시각화기록
created: 2025-06-28
updated: 2025-06-28
tags: [혼동행렬, XGBoost, 시각화, 오류해결, 정신건강]
---

## ✅ 목적
- 학생/직장인 그룹별로 우울 여부 예측 정확도를 시각화하기 위해 XGBoost 모델의 혼동행렬을 생성
- 포트폴리오 제출용 시각 자료이므로 시각적 일관성과 가독성 강화 필요

---

## ✅ 실행 단계
1. 전처리 완료된 `mental_train_preprocessed.csv` 로딩
2. `group`별 데이터 분리 및 `get_dummies` 적용
3. 저장된 모델(`xgb_students_model.pkl`, `xgb_professionals_model.pkl`) 로드
4. 예측 수행 및 혼동행렬 계산
5. 시각화 커스터마이징:
   - 커스텀 색상: `#3d4e62` 계열 톤
   - 배경 대비로 숫자 색상 자동 조절(luminance 기반)
   - 한글 깨짐 대응(OS 분기 후 폰트 설정)

---

## ✅ 오류 로그 및 해결

### ❌ 오류 내용
ValueError: 'colors' must be a string or Colormap. Got ['#3d4e62', '#E19992', '#D54344']

- 원인: `sns.color_palette()`는 list 반환 → `matplotlib`의 `cmap` 인자로 부적합
- 해결: `sns.light_palette("#3d4e62", as_cmap=True)` 사용하여 Colormap 생성

---

### ❌ 한글 폰트 깨짐
- 원인: 기본 폰트가 한글 미지원
- 해결: 시스템 분기하여 아래처럼 설정
```python
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'


✅ 결과 요약 및 해석
🎓 학생 그룹
TP: 15,260 / TN: 10,050 / FP: 1,515 / FN: 1,076

정확도: 약 91.9%

우울(Yes) 케이스 예측도 우수 → 상담 유도 기반 시스템 적합

💼 직장인 그룹
TP: 7,226 / TN: 102,605 / FP: 963 / FN: 2,005

정확도: 약 97.3%

우울(Yes) Recall 낮음 → 보수적 예측, 놓침 위험 존재

✅ 결과 이미지
![[confusion_students.png]]

![[confusion_workers.png]]

✅ 후속 연계
[[03_shap_분석_및_모델재학습결정.md]]에서 Feature Importance 및 우울 예측 기준 해석 연계 예정