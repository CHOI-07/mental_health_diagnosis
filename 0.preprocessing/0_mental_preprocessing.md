# 0. 데이터 전처리 로그

## 주요 처리 항목

- NaN 및 공백값 처리
  - 수치형: 평균값 또는 0 대체
  - 범주형: factorize 또는 수동 매핑
- Sleep Duration 이상치 제거 (IQR 기준)
  - 이유: 비현실적 값 존재 → 평균 및 분산 왜곡 방지
  - 기준: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
- 문자형 매핑
  - Gender, Degree, Family History, Suicidal Thought, Working/Student 등
- factorize 적용
  - City, Profession, Dietary Habits
- 파생 변수 생성
  - Total Pressure: Academic + Work Pressure 평균
  - Total Satisfaction: Study + Job Satisfaction 평균

## 시각화 저장 파일

- `missing_comparison.png`: 결측치 처리 전후 시각화
- `dtype_comparison.png`: 타입 변환 전후 확인

## 출력 파일

- `mental_train_preprocessed.csv` : 모델 학습용 전처리 결과

## 실행 로그 요약

```bash
> python 0_mental_preprocessing.py

이상치 제거 기준: 0.00 ~ 6.00
제거된 수면시간 이상치: 3,872개
최종 저장: mental_train_preprocessed.csv
