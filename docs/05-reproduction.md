# 05. 재현 방법

[프로젝트 홈](../README.md) / [단계별 기록](02-experiment-journey.md)

## 가장 쉬운 경로: Kaggle 노트북

[kaggle_notebook/](../kaggle_notebook/README.md)의 노트북은 이 저장소 없이 혼자 실행되며, 두 모드가 있다.

* **replay (기본):** 토큰 없이 CPU로 몇 초. 멤버 4개의 저장된 OOF·test 확률([kaggle_dataset/](../kaggle_dataset/README.md), 라벨 없음)을 불러와 fold가 다시 만든 fold와 같은지 확인하고, `train.csv` 로 nested 정확도를 다시 계산한 뒤 스태커를 학습한다. 출력 파일은 실제 제출 파일과 해시가 같다.
* **train:** Kaggle Secret `TABPFN_TOKEN`, GPU, 인터넷이 있으면 멤버 4개를 처음부터 학습한다. T4 기준 약 1.5–2.5시간.

로컬(RTX 4070 Ti SUPER)에서 train 모드를 처음부터 다시 돌린 결과, frozen 멤버는 기록과 완전히 같았고(0.826182) 파인튜닝 멤버는 ±0.0012 안에서 달랐다. nested 스택은 0.8294 / 0.8309로 기록(0.8291 / 0.8311)과 같은 수준이고, 생성된 제출 파일은 실제 제출 파일과 각각 28행, 43행(4,277행 중) 달랐다. 정확한 재현은 replay 모드가, 레시피 재현은 train 모드가 보여 준다.

## 로컬 환경

| 항목 | 사용한 버전 |
|---|---|
| OS / GPU | Windows 10, RTX 4070 Ti SUPER 16GB |
| Python | 3.12 (`uv` 로 관리, `uv.lock` 고정) |
| 주요 패키지 | numpy 2.5, pandas 3.0, scikit-learn 1.9, catboost 1.2.10, xgboost 3.4, torch 2.14.1+cu126, tabpfn 9.1.0, tabicl 2.2.0, pytabkit 1.7.3 |

정확한 목록은 [environment-snapshot.json](evidence/environment-snapshot.json)과 `uv.lock` 에 있다.

```powershell
uv sync
uv run python scripts/check_env.py
uv run pytest
```

TabDPT와 TabSTAR는 huggingface-hub 버전 충돌 때문에 `.venv-side` 에, AutoGluon은 `.venv-autogluon` 에 따로 설치했다 (둘 다 저장소에 포함하지 않음). 최종 모델에는 필요 없다.

## 데이터

공식 데이터는 포함하지 않는다. 대회 규칙에 동의한 계정으로 내려받는다.

```powershell
kaggle competitions download -c spaceship-titanic -p data\raw
```

압축을 풀어 `data/raw/train.csv`, `test.csv`, `sample_submission.csv` 를 둔다. 해시는 [data-fingerprints.json](evidence/data-fingerprints.json)과 같아야 한다. 모든 실행 결과(`reports/*_metrics.json`)에도 원본 파일 해시가 기록되어 있다.

## TabPFN 토큰

TabPFN v3.5 가중치는 Prior Labs 계정의 토큰과 라이선스 동의가 필요하다. 토큰은 사용자 환경변수 `TABPFN_TOKEN` 으로만 두고, 저장소에는 넣지 않는다.

```powershell
$env:TABPFN_TOKEN = [Environment]::GetEnvironmentVariable("TABPFN_TOKEN", "User")
$env:TABPFN_NO_BROWSER = "1"
```

## 주요 단계 재현

모든 러너는 설정·원본 데이터·코드 해시로 지문을 만들고, 같은 지문의 결과가 있으면 다시 학습하지 않는다. fold 배정은 `data/processed/sgkf_5_seed<seed>.csv` 에 고정된다.

| 단계 | 명령 |
|---|---|
| CatBoost 정직한 조기종료 screen | `uv run python scripts/run_hm09_stopping.py` (fixed300 대조군), `uv run python scripts/run_feature_screen.py --spec configs/hm10_spec.json` |
| TabPFN v3.5 vs CatBoost | `uv run python scripts/run_feature_screen.py --spec configs/ha12_spec.json --workers 1` |
| fold 안 파인튜닝 | `uv run python scripts/run_feature_screen.py --spec configs/ha23_spec.json --workers 1` |
| 파인튜닝 후 refit | `uv run python scripts/run_feature_screen.py --spec configs/ha24_spec.json --workers 1` |
| nested 스태커 screen / confirm | `uv run python scripts/run_ha27_stacker.py`, `uv run python scripts/run_ha27_confirm.py` |
| 100에폭 파인튜닝 + 추론 변형 | `uv run python scripts/tabpfn_ftbudget_runner.py A1 42 123 2026` |
| 멤버 추가·교체 스택 screen | `uv run python scripts/run_stack_member_screen.py <item> <member> [old=new ...]` |
| 제출 파일 생성 | `uv run python scripts/make_stack_submission.py <name> <member> ...` |

`tabpfn_ftbudget_runner.py` 는 변형 파일을 `<name>_seed<s>_<variant>` 로 쓴다. 스택 screen 전에 `<name>_<variant>_seed<s>` 로 복사해야 한다 ([handoff.md](../handoff.md)의 A 섹션 참고).

## 보고서 자산

```bash
uv run python tools/build_report_assets.py          # docs/assets/*.png 차트 (커밋된 증거만 읽음)
npm ci --prefix tools/diagram-renderer               # 흐름도 렌더러 (Viz.js + sharp)
npm run render --prefix tools/diagram-renderer       # docs/diagrams/*.dot → docs/assets/*.svg, *.png
python tools/verify_publication.py                   # 해시, 제출 형식, 문서 링크 확인
```

`tools/snapshot_evidence.py` 는 공개 직전에 한 번 실행해 제출 기록(Kaggle API 읽기), 데이터 지문, 환경 정보, 제출 파일 해시를 만들었다. Kaggle 인증과 원본 데이터가 있어야 한다.

## 재현의 한계

* TabPFN 파인튜닝은 GPU 커널의 비결정성 때문에 장비가 바뀌면 OOF가 조금씩 달라진다. 같은 장비에서는 지문 캐시로 기록된 결과를 그대로 다시 읽는다.
* OOF 예측 파일과 모델 가중치는 저장소에 넣지 않았다. 스택 결과를 다시 계산하려면 멤버들을 먼저 학습해야 한다.
* 일부 러너 이름에는 옛 슬롯 이름(`hm…`, `Main`)이 남아 있다. 러너 파일의 해시가 캐시 지문에 들어가므로 이름을 바꾸지 않았다.
