title: 3_shap_analysis_log
created: 2025-06-24
updated: 2025-06-24
tags:
  - SHAP
  - 실행로그
  - 오류정리

목표:
  - 학습된 XGBoost 모델에 대해 학생/직장인 그룹으로 나누어 SHAP 분석 실행
  - 주요 변수 시각화를 통해 모델 해석 가능성 확보

실행_흐름:
  - 전처리된 mental_train_preprocessed.csv 불러오기
  - 타겟(Depression) 및 직군 분리
  - get_dummies() 적용 → 모델 학습 시와 동일한 feature 형태 구성
  - joblib으로 저장된 모델 로드
  - shap.Explainer() 실행 및 시각화 결과 저장

오류_및_원인_분석:
  - 오류명: Empty dataset at worker
    로그: "[23:00:34] WARNING: Empty dataset at worker: 0"
    원인: 그룹 분리 이후 X_students 또는 X_workers가 비어 있는 상태에서 SHAP 실행
    설명: 전처리 또는 조건 필터링 오류로 인해 SHAP 입력값이 빈 데이터프레임이었음

  - 오류명: IndexError: too many indices for array
    로그: "self.expected_value = phi[0, -1]"
    원인: SHAP 내부 결과 배열의 차원이 예상보다 작을 때 발생
    설명: 입력 feature 수 mismatch 또는 SHAP 실행 대상 데이터가 비정상적으로 구성됨

  - 오류명: KeyError: ['Name_XXX' ...] not in index
    로그: "KeyError: ['Name_Aariket', ...] not in index"
    원인: get_dummies() 실행 시 그룹별로 나눠 적용해 one-hot 컬럼이 달라짐
    설명: 모델 학습 시 사용한 feature들과 SHAP 분석 대상 feature들이 일치하지 않아 발생

  - 오류명: XGBoostError: shape mismatch
    로그: "Check failed: ... == chunksize * rows (32034916 vs. 44668404)"
    원인: SHAP 내부 booster 예측 시, 입력 feature matrix의 크기(shape)가 예상과 다름
    설명: SHAP에 들어간 데이터가 모델이 기대하는 feature 순서/형태와 맞지 않음

해결_및_조치_요약:
  - get_dummies()는 전체 데이터셋에 일괄 적용한 후 그룹 분리 방식으로 재구성
  - SHAP 실행 전 X_students, X_workers에 대해 모델 feature 순서 맞추기 필수
  - 기존 저장된 모델들은 feature shape mismatch로 인해 해석 불가
  - 따라서 모델 재학습 필요 (문서: [[03_shap_분석_및_모델재학습결정.md]])
  - 재학습 후 1_model_retrain.py 기준으로 SHAP 분석 재실행 예정
