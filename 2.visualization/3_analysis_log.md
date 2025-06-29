error_summary:
  error_message: "ValueError: Length mismatch: Expected axis has 4 elements, new values have 3 elements"
  cause:
    - reset_index()를 사용하면 기존 인덱스가 컬럼으로 추가됨
    - groupby 결과나 인덱스 구조에 따라 컬럼 개수가 3개보다 많아질 수 있음
    - plot_df.columns = [idx_name, target, 'ratio']처럼 컬럼 이름을 3개로 강제 지정하면
    - 실제 컬럼 개수와 지정하려는 이름 개수가 달라서 ValueError 발생
  example_code:
    before: |
      plot_df = plot_series.reset_index()
      plot_df.columns = [idx_name, target, 'ratio']  # 여기서 에러 발생 가능
    after: |
      plot_df = plot_series.reset_index()
      if len(plot_df.columns) > 3:
          plot_df = plot_df.iloc[:, -3:]
      plot_df.columns = [idx_name, target, 'ratio']
  additional_notes:
    - 인덱스가 'Name' 등 기존 컬럼과 겹치면 reset_index()에서 또 다른 ValueError가 발생할 수 있음
    - 이 경우 인덱스 이름을 임시로 바꿔주는 처리가 필요함

   
```yaml
project_log:
  title: "정신건강 자가진단 데이터 분석 실험 로그"
  date: "2024-06-XX"
  author: "hee"
  environment:
    os: "macOS (darwin 24.5.0)"
    python: "3.10"
    packages:
      - pandas
      - matplotlib
      - seaborn
      - numpy
  data:
    train_file: "0.preprocessing/mental_train_preprocessed.csv"
    main_columns:
      - Depression (0/1)
      - Depression_Label (No Depression/Depression)
      - core_cols: ['Age', 'Sleep Duration', 'Satisfaction', 'Pressure', 'Financial Stress', 'Work/Study Hours']
  code_version: "최종"
  main_analysis_and_visualization:
    - 전체 변수 간 상관관계 히트맵 (임계치 이상만 숫자 표시, x축 45도)
    - Spearman 상관관계 히트맵 (vlag, x축 45도)
    - Depression 그룹별 핵심 변수 평균 비교 (차이 큰 순, 수치 라벨)
    - 타겟 분포 도넛 파이차트 (비율+개수, 글씨 검정, 조각 분리)
    - 유의미한 범주형 변수별 Depression 비율 (샘플 5개 미만 제외)
    - 주요 수치형 변수 박스플롯 (Depression 그룹별, 파일명 일관화)
  errors_and_solutions:
    - name: "ValueError: Length mismatch: Expected axis has 4 elements, new values have 3 elements"
      cause: "reset_index() 후 컬럼 개수와 columns 지정 개수 불일치"
      solution: "plot_df.columns 지정 전 컬럼 개수 확인, 필요시 마지막 3개만 사용"
    - name: "ValueError: cannot insert Name, already exists"
      cause: "reset_index() 시 인덱스 이름이 기존 컬럼과 중복"
      solution: "인덱스 이름이 'Name'이면 임시로 변경"
    - name: "파이차트 글자 잘림"
      cause: "figsize 부족, bbox_inches 미사용"
      solution: "figsize 키우고, bbox_inches='tight'로 저장"
    - name: "상관관계 히트맵 숫자 과다"
      cause: "annot=True로 모든 값 표시"
      solution: "임계치(|r|>0.3) 이상만 표시"
    - name: "의미 없는 범주형 변수(Name 등) 자동 포함"
      cause: "범주형 변수 자동 추출 시 id, Name 등 포함"
      solution: "exclude_cols로 제외"
  experiment_log:
    - "데이터 전처리 및 통합 변수 생성"
    - "Depression 컬럼 라벨화 및 target 변수 변경"
    - "상관관계 히트맵, Spearman 히트맵, 평균 비교, 파이차트, 범주형/수치형 변수별 시각화 일괄 실행"
    - "시각화 파일명 일관화 및 저장"
  reproducibility:
    - "코드 전체 실행 시 동일한 결과 재현 가능"
    - "시각화 파일명, 저장 경로, 분석 방식 모두 코드에 명시"
    - "분석 환경(패키지, OS) 명시"
  results_and_summary:
    - "Depression 그룹 간 Age, Financial Stress, Work/Study Hours 등에서 뚜렷한 평균 차이"
    - "타겟 분포 불균형(Depression:No Depression ≈ 8:2)"
    - "유의미한 상관관계만 히트맵에 숫자로 표시, 시각적 인사이트 강화"
    - "범주형 변수 중 의미 없는 변수 자동 제외, 신뢰도 낮은 범주(샘플 5개 미만) 제외"
    - "박스플롯, 바플롯, 파이차트 등 시각화 결과 모두 저장 및 보고서 활용 가능"
  notes_and_next_steps:
    - "추가 변수 엔지니어링, 모델링, 외부 데이터 결합 등 확장 가능"
    - "시각화 스타일/임계치/분석 방식은 목적에 따라 유연하게 조정"
```
