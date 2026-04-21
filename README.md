# Leaf 작업 폴더

`Trait-Based-Essay-Feedback/`는 LEAF 데이터셋 전용 작업 폴더입니다.
이 폴더만 다른 위치로 옮겨도 실행할 수 있도록 구성되어 있습니다.

구성:
- `data/raw/leaf--main/`: 새로 받은 LEAF 원본 split CSV와 rubric 파일
- `data/raw/leaf--main.zip`: 원본 ZIP 사본
- `data/leaf_merged.feedback4_only.csv`: 기존 4개 feedback dimension 기준 호환 CSV
- `data/leaf_merged.csv`: 기존 점수 컬럼 + 새 10개 trait를 합친 통합 CSV
- `scripts/prepare_leaf_dataset.py`: ZIP에서 raw/merged 데이터를 재생성하는 준비 스크립트
- `legacy_data/filtered_leaf3000.csv`: 예전에 루트 레포에 있던 LEAF 관련 보조 데이터
- `runtime/promptaes2/`: 실행 스크립트가 사용하는 로컬 PromptAES2 런타임
- `.venv/`: `uv`로 만든 로컬 가상환경
- `results/`: LEAF 전용 체크포인트와 학습 결과 저장 경로
- `requirements.txt`: 이 작업 폴더에서 필요한 Python 패키지 목록
- `_common.sh`: 공통 실행 함수
- `setup_uv.sh`: `uv` 가상환경 생성 및 의존성 설치
- `run_pretrain.sh`: 10-trait 사전학습 예시
- `run_holistic.sh`: 10-trait embedding 기반 holistic 학습 예시
- `run_trait_score.sh`: 10개 trait-score 학습 예시
- `run_all.sh`: pretrain → holistic → trait-score 순차 실행
- `사용가이드.md`: 실행 순서와 동작 방식을 정리한 안내 문서

데이터셋 정리 방식:
- root split (`train/dev/test.csv`)의 `trait_1`~`trait_10`을 가져옵니다.
- `LEAF-Feedback-dimension/`의 4개 dimension과 `overall`을 가져옵니다.
- 같은 `(ID, split)` 기준으로 합쳐서 `data/leaf_merged.csv`를 만듭니다.
- 기존 파이프라인 호환을 위해 `total_score`는 4개 feedback dimension의 합으로 유지합니다.
- `feedback_dimension_overall`은 별도 컬럼으로 보존합니다.

추가된 10개 semantic trait:
- `grammar_accuracy`
- `appropriateness_of_word_use`
- `elasticity_of_sentence_expression`
- `appropriateness_of_structure_within_a_paragraph`
- `adequacy_of_inter_paragraph_structure`
- `consistency_of_structure`
- `appropriateness_of_portion_size`
- `clarity_of_topic`
- `specificity_of_explanation`
- `creativity_of_thought`

권장 trait 그룹:
- 표현: `grammar_accuracy,appropriateness_of_word_use,elasticity_of_sentence_expression:3`
- 조직: `appropriateness_of_structure_within_a_paragraph,adequacy_of_inter_paragraph_structure,consistency_of_structure,appropriateness_of_portion_size:4`
- 내용: `clarity_of_topic,specificity_of_explanation,creativity_of_thought:3`

이렇게 묶는 이유:
- 표현 그룹은 문법, 어휘, 문장 표현 품질을 한 축으로 묶습니다.
- 조직 그룹은 문단 내부 구조, 문단 간 연결, 전체 구조 일관성, 분량 배분을 한 축으로 묶습니다.
- 내용 그룹은 주제 선명도, 설명의 구체성, 사고의 창의성을 한 축으로 묶습니다.
- 현재 실행 스크립트는 이 3그룹을 먼저 homo로 합친 뒤, 그룹 간 hetero interaction으로 holistic/trait-score를 학습합니다.

주의점:
- 원본 `rubrics.json`은 형식이 깨져 있어서, 정리된 버전은 `data/raw/leaf--main/rubrics.normalized.json`에 따로 저장합니다.
- raw `trait_10`에는 값 `0`이 4개 있으며, 정규화 컬럼 `creativity_of_thought`에서는 이를 결측치로 처리합니다.
- 현재 실행 스크립트는 `--predefined_split_column split`을 전달해서 `split` 컬럼의 `train/dev/test`를 공식 분할로 그대로 사용합니다.
- `split_by_column`은 group-by 용도이고, 공식 train/dev/test 분할에는 `--predefined_split_column`을 써야 합니다.
- `_common.sh`에 LEAF 공용 trait 목록, 3그룹 문자열, `class_balance_mode=loss_and_sampler`, `model_variant=canonical_moe`, `warmup_epochs=3` 기본값을 모아뒀습니다.
- holistic/trait-score의 hetero 단계는 예전 pairwise relation 대신 `GroupInteractionEncoder`로 3개 group token을 직접 contextualize합니다.
- 실행 스크립트는 `uv` 가상환경의 Python과 로컬 `runtime/`를 사용하므로, 상위 PromptAES2 레포 위치에 의존하지 않습니다.
- `uv`가 아직 없으면 먼저 설치해야 합니다. 예: `python3 -m pip install uv`
- LEAF의 `total_score`는 여전히 `12~14` 구간에 많이 몰려 있습니다. 새 10-trait 데이터는 supervision은 늘리지만, holistic 점수 불균형 자체를 없애지는 않습니다.
