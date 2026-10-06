# 공유 노트북 미실행 아이디어 조사 (점수순 상위 100, 약 0.81–0.83 구간)

조사일: 2026-10-05, Agent A (Claude Code). 기존 카탈로그(`clean_notebook_catalog.md`, 18개 점검)에
이어 점수순 목록 `kernels-score.json` 100개 중 **아직 점검하지 않은 65개 소스**와, 캐시만 되어 있던
eyefulye 2개, 기존에 최종 경로만 보고 제외했던 arunklenin의 FE 부분을 정적으로 읽었다.
코드는 실행하지 않았다. 소스는 `.kaggle-research/spaceship-titanic/kernel-archives/<owner>__<slug>/`
(Git 제외, 신뢰하지 않는 텍스트)에 있다. CLI 목록에는 점수가 없으므로 "0.81–0.83"은 점수순 상위
100개(1위 0.82137 표기, 100위 "81.1%" 표기) 범위로 해석했다. 개별 노트북 점수는 확인하지 않았다.

## 중복 정리

토큰 5-gram Jaccard > 0.6으로 묶으면 65개 → 약 61개. 큰 복제 묶음:

- larjeck 계열 11개(bunny11, haramasayosi, hirotakaaa, javinlim-notebook-1, leeruyuan, masatakawatanabe,
  moriyuuki, nakaitomoki, tamakii, yuntechimb11123020, +amirmohammadparvizi·chiragmohangupta 재포장).
  Viktor/twinpilgrim XGBoost 계보와 같은 파라미터(730 trees, depth 5, lr 0.0667) + 전체 train SMOTE.
- opamusora 계열 5개(nutchacharoenwong ×3, kullanutpadtaku), deepakkaura와 거의 같음.
- 지수가중 CSV 앙상블 계열(rm1000, rizkykiky, saraswatitiwari, areebsyed237-ensemble, chiejuwonsfx, yisberh).
- dmitryshekhov 계열(desalperterson, areebsyed237-catboost-thresholding).

## 제외 (프로젝트 규칙 위반)

| 이유 | 노트북 |
| --- | --- |
| 다른 제출 CSV로 최종 예측(OR/투표/LB 가중) | rm1000, rizkykiky, saraswatitiwari, yisberh, areebsyed237-ensemble, chiejuwonsfx, meenalsinha, ravi20076-submission-v2, yuntechimb11123055 (외부 parquet 특성 + pseudo-label + 외부 CSV OR) |
| 하드코딩된 4,277비트 예측 덮어쓰기 + 자동 제출 | eyefulye ×2 (bsthere와 같은 `use_best_public_override` 방식) |
| 외부 서비스·외부 데이터 사본 | gautamshah2002 (Headjack 원격 변환, 소스에 자격증명 포함), kevineffendy (gist에서 데이터 로드) |
| CV 전 타깃 사용 | opamusora 계열·deepakkaura (클래스별 Age 평균), eu1234 (Deck 전이율), fatalxs·fishpain·wamiottowang·seetohzijie (TargetEncoder), rayenghali023 (타깃 포함 LOF), wajahat1064 (WOE·클러스터 타깃평균) |
| CV 전 SMOTE | larjeck 계열, win0sexp, muhammadfiqriadam, rayenghali023, jimmyyeung |

나머지 대부분은 정상 직접 학습이지만 그룹 비분리 CV, 채점 fold 조기 종료, 같은 holdout에서 임계값 선택 등
검증 한계가 있다(D-D-018, D-D-026과 같은 유형).

## 아직 실행하지 않은 아이디어 → plan 반영

target-free 커버리지는 `scripts/audit_notebook_ideas.py` → `reports/notebook_idea_coverage.txt`.

| 아이디어 | 출처 노트북 | 근거(공식 특성만) | plan |
| --- | --- | --- | --- |
| 그룹 내 일치 예측으로 불확실 행 보정(사후처리) | eyefulye L305 (미검증), meenalsinha L180 | 그룹 단위 SGKF라 OOF에서 정직하게 평가 가능, 기존 OOF만으로 시험 가능 | H-A-01 |
| Surname을 CatBoost 범주형으로(fold 내부 ordered TS) + 그룹 내 성씨 반복 수 | khinfoong, eyefulye, pasuvulasaikiran, kacperrabczewski, opamusora 계열 | test 행의 89.9%가 train에 있는 성씨, train 행의 90.4%가 다른 그룹에 같은 성씨 | H-A-02 |
| 위치 특성: GroupId 수치, deck별 GroupId→CabinNum 회귀 보완, Cabin 점유 수 | desalperterson L246, areebsyed237 L181, eu1234 L219/L344, phattharakit, khinfoong | deck별 corr(GroupId, CabinNum) 0.980–1.000, Cabin은 그룹 간 공유 0/9,825 | H-A-03 |
| 역방향 CryoSleep 규칙 + 그룹 일치 Cabin/VIP 채우기 | larjeck 계열 L108·L114, win0sexp, seongwookim, jimmyyeung | CryoSleep 결측+완전 0지출 train 87/test 36, P(Cryo)=0.859(13세 이상 0.964); Cabin 87/31, VIP 86/36 | H-A-04 |
| 비CatBoost 다양성 학습기(YDF BEST_FIRST_GLOBAL, CatBoost Lossguide, 임베딩 MLP) + 중첩 LR 스태킹 | taedubb, yueyingwen, kulibayev, pasuvulasaikiran, mohamedabdalla33, cv13j0, lemonmeringueee, aegistea(AutoGluon) | 공개 CV는 오염, 효과 미확인 | H-A-05 |

## 검토했지만 plan에 넣지 않은 것

- log/sqrt/Box-Cox/Yeo-Johnson/Quantile 변환, 연령 구간 재분할, 지출 상한: 단조 변환이라 트리 분할을 바꾸지 않음.
  NN/선형 멤버를 만들 때만 H-A-05 안에서 사용.
- KNN/Iterative/LGBM 대체: CatBoost가 결측을 직접 다루고, 같은 계열의 그룹 채우기(v2_group)가 실패(D-D-014).
- 저카디널리티 범주 target/WOE/count 인코딩, KMeans 클러스터: CatBoost 내장 처리와 중복, 공개 구현은 누출.
- Cabin/HomePlanet/Destination 결측 지시자: 이미 `__MISSING__` 범주로 모델에 들어감.
- IsolationForest/LOF 이상치 제거, SMOTE, `scale_pos_weight`: 클래스가 균형이고, 임계값 이동은 H-D-06 범위.
- 테스트 pseudo-label: 정답 누출은 아니나 OOF로 정직하게 평가하기 어렵고 규칙상 사전 고지 필요 → 보류.
- 그룹 Destination 채우기: 다인 그룹 중 Destination 일치 그룹은 42.9%뿐이라 신뢰 낮음.
- 고정 drop 목록(eli5/RFECV/SFS): H-D-03(특성 제거)와 중복.
