# 08. 연구에 사용한 스킬

[프로젝트 홈](../README.md) / [작업 방식과 에이전트](04-workflow-and-agents.md)

에이전트 "스킬"은 에이전트가 특정 작업을 할 때 따르는 지침과 도구 묶음이다. 이 연구에서 실제로 사용한 스킬과, 각 스킬이 무엇을 했는지 남긴다. 스킬은 모델에게 답을 알려 주는 것이 아니라 **작업 절차와 도구 사용법**을 정한다. 예측은 모두 로컬에서 학습한 모델이 만들었다.

## 요약

| 스킬 | 만든 사람 | 사용한 호스트 | 역할 | 공개 시점 스냅샷 |
|---|---|---|---|---|
| [research-orchestrator-skill](https://github.com/TaeyanG4/research-orchestrator-skill) | TaeyanG4 (MIT) | Claude Code, Codex, ChatGPT | 연구 규약 전체: 가설 큐, 발견 기록, 교차 검증, 인계 로그, 일관성 검사 | [skills/research-orchestrator-skill/](skills/research-orchestrator-skill/) |
| [kaggle (unofficial)](https://github.com/shepsci/kaggle-skill) v3.0.1 | shepsci (MIT) | Codex | 공개 노트북 수집, 제출 전 파일 검사와 제출 기록, 리더보드 확인 | [skills/kaggle/](skills/kaggle/) |
| `update-config` | Claude Code 내장 | Claude Code | 알림 설정 검토 (최종적으로 상시 훅은 남기지 않음) | 해당 없음 |

## research-orchestrator-skill

이 저장소의 루트에 있는 네 파일이 이 스킬의 산출물이다.

| 파일 | 스킬이 정한 형식 |
|---|---|
| [agents.md](../agents.md) | 고정 규칙. 슬롯 이름(A, B, C …), 읽는 순서, 무결성·검증 규칙, 우선순위 식 |
| [plan.md](../plan.md) | 에이전트별 가설 큐. 항목마다 `Sources / Hypothesis / Evidence / Improvement`, 점수 6개, `Priority`, 자원(CPU·GPU) |
| [discoveries.md](../discoveries.md) | 실험마다 발견 하나. `Host`, `Cross-check` 상태(PENDING → VERIFIED / HOLD / CHALLENGED), 다른 호스트의 `Reviews` |
| [handoff.md](../handoff.md) | 공유 상태(현재 챔피언), 슬롯별 이어받기 상태, 완료 이벤트 로그 |

이 스킬 덕분에 생긴 습관들이다.

* **중복 검사:** 새 가설을 추가하기 전에 실패 기록까지 검색했다. Playground 기법 8개 중 6개, 사용자가 제안한 pseudo-labelling은 실험 없이 기존 발견으로 정리됐다.
* **실패도 기록:** 실험이 끝나면 발견 → 로그 → 후속 가설 → 큐에서 제거를 한 번에 처리했다. 78개 발견 중 대부분이 "does not hold under …" 형식의 실패 기록이다.
* **교차 검증:** 한 호스트의 발견은 다른 호스트가 한 번 검토한다. ChatGPT가 등록한 재검토 항목으로 D-D-021이 CHALLENGED로 뒤집혔다.
* **이어받기:** 세션이 끊기거나 모델이 바뀌어도 `handoff.md` 의 상태와 명령으로 같은 작업을 이어 갔다.
* **일관성 검사:** 스킬의 `scripts/check_project.py` 로 네 파일의 형식을 수시로 검사했다.

### 버전

스킬은 연구 도중에도 여러 번 갱신됐고, 그때마다 프로젝트 문서를 새 규격에 맞췄다 (handoff.md의 "Migration" 이벤트).

| 시점 (현지 시각) | 내용 |
|---|---|
| 10-05 21:08 | v7.1.0으로 이전. 스킬을 Claude Code와 Codex에 설치하고 네 파일을 새 형식으로 다시 씀 |
| 10-05 23:15 | 호스트 기반 형식과 교차 검증으로 변환 (`H-D-NN` / `D-D-NNN` ID, `Host`, `Cross-check`) |
| 10-06 01:55 | A부터 시작하는 슬롯 이름, host = 플랫폼, 같은 호스트 항목 인계 규칙 반영 |
| 10-06 02:08 | 01:59 업데이트 적용: 단일 행 필드, 이벤트 형식, `Current best` 공유 상태, 일관성 검사. 이후 Codex 사본 마지막 동기화 02:21 |
| 10-06 18:12 | 연구 종료 직후 최신 버전 배포 (`Users`, `Current user` 필드 추가). 이 저장소의 `handoff.md` 는 이 형식으로 아직 옮기지 않았다 |

[skills/research-orchestrator-skill/](skills/research-orchestrator-skill/)에는 **연구 대부분에 적용된 02:21 버전**을 그대로 보관했다 (SKILL.md SHA256 앞자리 `eebe653dc2c48253`). 최신 버전은 [GitHub 저장소](https://github.com/TaeyanG4/research-orchestrator-skill) (공개 시점 커밋 `f5d9a40`)에 있다.

## kaggle (shepsci/kaggle-skill)

Kaggle 웹·API 작업을 묶은 비공식 스킬이다. 초기 Codex 슬롯이 사용했다.

| 사용 | 근거 |
|---|---|
| 점수 순 100개, 추천 순 30개 공개 노트북 목록과 소스 수집 (읽기 전용, 실행하지 않음) | handoff.md "Shared notebook provenance and validation audit", D-D-017 |
| 첫 제출의 파일 검사와 제출·점수 기록 | 로컬 `.kaggle-skill/ledger.jsonl` (제출 56847813, 해시와 `file_check: passed`) |
| 리더보드 스냅샷 확인 | 로컬 `.kaggle-skill/leaderboard/` |
| 스킬의 MCP 읽기 경로 실패 기록 | D-D-020 (CLI는 정상, MCP 샘플 조회만 실패) |

이후 Claude Code는 같은 작업을 Kaggle CLI 2.2.4로 직접 했다 (제출, 제출 기록 조회, 커널 목록). 스냅샷에는 v3.0.1의 `SKILL.md` 와 MIT 라이선스만 넣었다. 스킬이 참조하는 모듈과 스크립트는 [원본 저장소](https://github.com/shepsci/kaggle-skill) (커밋 `a47bff9`)에 있다.

## 스킬이 아닌 도구

| 도구 | 쓰임 |
|---|---|
| Claude Code 서브에이전트 (general-purpose) | Playground Series 상위 솔루션 조사(D-A-034), TabPFN 파인튜닝 가설 수집(H-A-39~42). 둘 다 읽기 전용, 결과는 사람이 검토한 뒤 플랜에 반영 |
| 웹 조회 (WebFetch, 공개 페이지 리더) | 공개 노트북 점수 출처 확인 (D-A-041), 참고 문헌 |
| 내장 브라우저 | 공개 후 GitHub 페이지 렌더링 확인 |
| `scripts/alert.py` | 승격된 모델이 나올 때만 알림음 (상시 훅 없음) |
