# 0. 데이터 전처리 및 파생 변수 생성

## 주요 처리 내용
- `NaN`, 공백값 처리
- 문자형 변수 숫자형 매핑 (Gender, Degree 등)
- `Sleep Duration`, `City`, `Profession` 등 factorize
- 학업/직무 압력 → `Total Pressure` 생성
- 학업/직무 만족도 → `Total Satisfaction` 생성

## 전처리 전후 비교 시각화
- `missing_comparison.png` : 결측치 처리 여부 확인
- `dtype_comparison.png` : object → numeric 변환 효과 확인

## 산출물
- `mental_train_preprocessed.csv` : 모델 학습용

## 비고
- 직업군에 따라 학업/직무 중 하나만 응답 → 평균 처리 방식 사용
