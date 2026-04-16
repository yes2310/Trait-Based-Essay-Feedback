# Leaf 작업 폴더

`leaf_workspace/`는 LEAF 데이터셋 전용 작업 폴더입니다.
이 폴더만 다른 위치로 옮겨도 실행할 수 있도록 구성되어 있습니다.

구성:
- `data/leaf_merged.csv`: PromptAES2에서 사용할 수 있도록 정규화한 LEAF CSV
- `data/leaf_merged_3class.csv`: trait 점수를 `1/2/3`으로 재매핑한 3-class LEAF CSV
- `legacy_data/filtered_leaf3000.csv`: 예전에 루트 레포에 있던 LEAF 관련 보조 데이터
- `runtime/promptaes2/`: 실행 스크립트가 사용하는 로컬 PromptAES2 런타임
- `.venv/`: `uv`로 만든 로컬 가상환경
- `results/`: LEAF 전용 체크포인트와 학습 결과 저장 경로
- `requirements.txt`: 이 작업 폴더에서 필요한 Python 패키지 목록
- `_common.sh`: 공통 실행 함수
- `setup_uv.sh`: `uv` 가상환경 생성 및 의존성 설치
- `run_pretrain.sh`: LEAF trait 사전학습 예시
- `run_holistic.sh`: LEAF holistic 학습 예시
- `run_trait_score.sh`: LEAF trait-score 학습 예시
- `run_all.sh`: pretrain → holistic → trait-score 순차 실행
- `사용가이드.md`: 실행 순서와 동작 방식을 정리한 안내 문서

정규화한 컬럼 매핑:
- `essay_text` -> `text`
- `overall` -> `total_score`
- `human_feedback_text` -> `human_feedback`
- `AI-augmented_feedback_text` -> `ai_augmented_feedback`

파이프라인에서 사용하는 LEAF trait:
- `alignment_with_topic`
- `spelling_grammar_style`
- `clarity_of_view_point`
- `arguments_supporting_details`

권장 trait 그룹:
- `alignment_with_topic,arguments_supporting_details:2`
- `clarity_of_view_point,spelling_grammar_style:2`

이렇게 묶는 이유:
- 1그룹은 주제 일치성과 근거 제시 품질을 함께 본다
- 2그룹은 주장 명확성과 표현 품질을 함께 본다
- 그룹이 2개 이상 있어야 hetero 상호작용이 의미 있게 동작한다

참고:
- 원본 CSV의 `split` 컬럼은 그대로 보존되어 있다
- 현재 실행 스크립트는 `data/leaf_merged_3class.csv`를 사용하고, trait 점수는 `1,2 -> 1`, `3 -> 2`, `4,5 -> 3`으로 재매핑되어 있다
- 현재 실행 스크립트는 `--predefined_split_column split`을 전달해서 `split` 컬럼의 `train/dev/test`를 공식 분할로 그대로 사용한다
- `split_by_column`은 여전히 group-by 용도이고, 공식 train/dev/test 분할에는 `--predefined_split_column`을 써야 한다
- 실행 스크립트는 `uv` 가상환경의 Python과 로컬 `runtime/`를 사용하므로, 상위 PromptAES2 레포 위치에 의존하지 않는다
- `uv`가 아직 없으면 먼저 설치해야 한다. 예: `python3 -m pip install uv`
- LEAF의 `total_score`는 `12~14` 구간에 많이 몰려 있어서, 드문 점수 구간은 holistic 분류가 불안정할 수 있다
- 루트 레포의 leaf 전용 설정과 문서는 제거했고, 이제 leaf 관련 자산은 이 폴더만 기준으로 관리한다
