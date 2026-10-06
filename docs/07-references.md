# 07. 참고 자료

[프로젝트 홈](../README.md)

## 대회

* [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) — Kaggle Getting Started 대회. 데이터는 대회 규칙에 따라 내려받으며 이 저장소에는 포함하지 않는다.

## 모델과 라이브러리

| 이름 | 사용한 곳 | 출처 |
|---|---|---|
| TabPFN v3.5 (`tabpfn` 9.1.0) | 최종 모델의 모든 멤버, 파인튜닝 (`FinetunedTabPFNClassifier`) | [Prior Labs TabPFN](https://github.com/PriorLabs/TabPFN), [문서](https://docs.priorlabs.ai/) |
| TabPFN 파인튜닝 연구 | 파인튜닝 예산·에폭 가설의 근거 | Rubachev et al., *On Finetuning Tabular Foundation Models*, [arXiv:2506.08982](https://arxiv.org/abs/2506.08982) |
| CatBoost | 1단계 챔피언 | [catboost.ai](https://catboost.ai/) |
| XGBoost, LightGBM, scikit-learn | 비교 모델, CV, 스태커 | [xgboost](https://xgboost.ai/), [lightgbm](https://lightgbm.readthedocs.io/), [scikit-learn](https://scikit-learn.org/) |
| TabICL | 비교 모델 | [soda-inria/tabicl](https://github.com/soda-inria/tabicl) |
| TabDPT | 비교 모델 | [layer6ai-labs/TabDPT-inference](https://github.com/layer6ai-labs/TabDPT-inference) |
| RealMLP, TabM, TabR (`pytabkit`) | 비교 모델 | [dholzmueller/pytabkit](https://github.com/dholzmueller/pytabkit) |
| xRFM | 비교 모델 | [dmbeaglehole/xRFM](https://github.com/dmbeaglehole/xRFM) |
| TabSTAR | 비교 모델 | [alanarazi7/TabSTAR](https://github.com/alanarazi7/TabSTAR) |
| Causilo | 비교 모델 | [nums-ai/causilo](https://github.com/nums-ai/causilo) |
| AutoGluon | 환경만 준비, 실험 미완료 | [auto.gluon.ai](https://auto.gluon.ai/) |
| Viz.js, sharp | 흐름도 렌더링 | [viz-js](https://github.com/mdaines/viz-js), [sharp](https://sharp.pixelplumbing.com/) |

## 검증 방법

* scikit-learn [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
* scikit-learn [Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
* scikit-learn [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

## 공개 노트북과 솔루션

공개 노트북은 실행하지 않고 코드만 읽었다. 예측 파일이나 코드 본문은 이 저장소에 포함하지 않았다.

| 자료 | 이 프로젝트에서의 쓰임 |
|---|---|
| [Viktor Taran - Space Titanic](https://www.kaggle.com/code/viktortaran/space-titanic) (Best Score 0.81833, V242) | 깨끗해 보이는 공개 최고점 기준. 소스를 보고 재구현해 정직한 OOF 약 0.805 확인 (D-A-014, D-A-041) |
| 점수 순 상위 공개 노트북 약 80개 | 아이디어 목록과 누출·override 감사 (D-D-017, D-D-018, D-A-001, D-A-003) |
| Kaggle Playground Series S4E10, S5E4, S5E8, S6E2, S6E3, S6E5, S6E8 상위 솔루션 글 | 아직 시도하지 않은 기법 조사. 잔차 부스팅(S4E10 1위), 메타 학습기 입력 확장(S5E4 1위), FM 멤버(S6E5 5위)를 시험 (D-A-034) |

## 작업 도구

* Codex, Claude Code, ChatGPT — 실험 코드 작성과 실행, 기록
* `research-orchestrator` 스킬 — 네 파일(agents / plan / discoveries / handoff) 기반 연구 규약
* Kaggle CLI 2.2.4 — 데이터 다운로드, 제출, 제출 기록 조회
* uv — Python 환경과 잠금 파일
