# Spaceship Titanic 공유 노트북 조사

조사일: 2026-10-05. 점수순 100개·투표순 30개 공개 노트북 목록을 검색하고,
18개 소스·메타데이터의 입력 파일, 타깃 사용, 검증, 최종 예측 경로를 정적으로 점검했다.
아래 6개는 **공식 데이터에서 직접 학습하는 정상 기법의 참고 목록**이다.
원문을 그대로 실행해도 검증 누출이 없다는 인증 목록은 아니다. 수정할 부분을 함께 적었다.

점수는 Kaggle 페이지의 **Best Score**이며 우리 프로젝트에서 재현한 점수가 아니다.
다운로드는 `kaggle kernels pull --metadata`의 최신 소스다. 최고 점수 버전과 최신 소스가
다를 수 있어 현재 소스에서 최고 점수가 재현된다고 단정하지 않는다.
다운로드한 코드는 실행하지 않았고, 제출 CSV·모델·외부 정답 데이터는 다운로드하지 않았다.

## 판정 기준

- 출처 불명의 고정 예측값 또는 테스트 정답으로 덮어쓰는 구현은 제외한다.
- 다른 노트북의 제출 CSV를 읽어 최종 예측을 만드는 구현은 이 프로젝트 규칙에 따라 제외한다.
  예측 앙상블 자체를 모든 대회에서 치팅이라고 단정하는 것은 아니다.
- 공식 train/test의 **특성만** 이용한 그룹·객실 관계 구성은 허용되는 방법이다.
  타깃을 사용하지 않는 기작과 검증 시 같은 정보가 제공된다는 점을 문서화해야 한다.
- 테스트 정답 누출과 검증 오염을 구별한다. 전체 데이터로 먼저 학습한 전처리,
  바깥 검증 전에 만든 타깃 기반 특성, 검증 결과를 보고 선택한 특성·임계값은 별도 문제다.
- 우리 적용에서는 PassengerId 그룹을 분리한 SGKF, fold 안에서 학습한 전처리,
  OOF 저장을 사용한다. 특성 선택·임계값·블렌드 학습은 내부 그룹 검증에서 결정한다.

## 점수와 방법별 참고 목록

| 노트북 / 작성자 | 표시 최고 점수 | 방법 | 원문 검증의 한계 |
| --- | ---: | --- | --- |
| [Space Titanic — Viktor Taran](https://www.kaggle.com/code/viktortaran/space-titanic) | **0.81833 · V242** | 동승 그룹 결측값 보완, 비용·CryoSleep 규칙, XGBoost, 특성 축소 | 전처리 후 일반 CV, 특성 선택 결과의 CV 재사용. Optuna를 켜면 fold 밖 SMOTE |
| [Optuna XGB — Mr.Tao](https://www.kaggle.com/code/guanlintao/0-814-optuna-xgb-space-titanic) | **0.81646 · V2** | 위 계열의 간소화, 비용·객실 특성, XGBoost, KMeansSMOTE | 전체 특성 전처리 후 CV. 최종 oversampling 모델의 독립 검증 없음 |
| [XGBoost Score 81.5 — twinpilgrim](https://www.kaggle.com/code/twinpilgrim/spaceshiptitanic-xgboost-score-81-5) | **0.81599 · V2** | 총지출·객실 deck/side, 간단한 결측값 보완, 특성 축소, XGBoost | 내려받은 소스에 CV 없음. train/test 공동 전처리는 타깃을 사용하지 않음 |
| [CatBoost · Increase Threshold — Dmitry Shekhov](https://www.kaggle.com/code/dmitryshekhov/catboost-increase-threshold-score-0-81) | **0.81318 · V41** | 그룹·객실·가족 특성, CatBoost, 확률 임계값 변경 | 같은 무작위 holdout으로 조기 종료와 임계값 선택·평가 |
| [81% · Models + Backward Feature Selection — Misael Ribeiro](https://www.kaggle.com/code/misaelcribeiro/81-models-backward-feature-selection) | **0.81271 · V8** | CatBoost + backward SequentialFeatureSelector | 전처리 후 일반 CV. 최종 특성 선택 성능의 별도 바깥 검증 없음 |
| [A complete guide — Samuel Cortinhas](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide) | **0.80874 · V47** | 규칙·그룹·객실 보완, CatBoost + LightGBM 평균 | 전체 특성 전처리 후 일반 CV, 테스트 양성 비율 0.519 가정으로 임계값 결정 |

상위 XGBoost 3개는 상당 부분 같은 계보이므로 독립적인 3개 성공 사례로 세지 않는다.
방법은 **작은 XGBoost / CatBoost 특성 선택 / CatBoost 임계값 / CatBoost·LightGBM 평균**으로 묶는다.
이 조사 범위나 점수가 정상 모델링의 전체 최고점·상한을 뜻하지 않는다.

## 소스에서 확인한 내용과 활용 방향

스냅샷은 `.kaggle-research/spaceship-titanic/kernel-archives/<폴더>/`에 있다.
`source_code.txt`는 코드 셀만 순서대로 합친 파일이며 아래 줄 번호의 기준이다.
위 6개 페이지는 Apache 2.0 라이선스를 표시한다. 작성자·원본 링크를 보존하며,
파일별 SHA256·의존성·점수 증거·판정은 `reports/clean_notebook_catalog.json`에 기록했다.

### Viktor Taran: 그룹 보완 + XGBoost

- 공식 train/test/sample만 읽고, 최종 `XGB_best.csv`를 모델 예측에서 만든다.
- PassengerId 앞부분인 동승 그룹으로 Cabin/VIP/HomePlanet/Destination을 보완한다.
  원문의 `Room`은 실제 객실 번호가 아니다. 첫 관측값을 채우므로 여러 객실·충돌 문제를
  해결하지 않는다. 우리 구현에서는 관측된 동료값의 일치 여부를 확인해야 한다.
- SimpleImputer·OHE를 전체 특성에서 학습한 뒤 CV를 수행한다(255줄 부근).
  permutation importance로 특성을 제거한 후 같은 데이터에서 CV를 비교한다.
- SMOTE는 351줄 부근에 있다. 현재 LGBM/XGB Optuna 스위치는 OFF라서 활성 초기 CV는
  SMOTE 전에 수행된다. Optuna 경로를 켜면 SMOTE 이후 CV가 되어 검증이 오염된다.
- 페이지 관측: V340/389, Public 0.81739, Best 0.81833 V242. 최신 소스와 V242의 일치는 미확인.

### Mr.Tao: 같은 계열의 변형

- 공식 입력만 사용하고 XGBoost 모델 예측으로 최종 결과를 만든다(333줄 부근).
  타깃은 모델 y로 분리하며 그룹 보완·imputer·OHE의 입력 특성에 넣지 않는다.
- KMeansSMOTE는 기존 CV 비교 뒤 최종 학습 전에 적용한다. 현재 활성 경로에 SMOTE 뒤
  CV가 없으므로 최종 모델의 CV가 검증됐다고 볼 수 없다.
- 현재 Optuna 예시는 주석 처리되어 있다. 제목의 0.814와 달리 실제 페이지 Best는
  0.81646 V2다. 페이지 버전은 V9/9이며 최고 점수 버전의 소스 일치는 미확인.

### twinpilgrim: 가장 작은 재구현 출발점

- 97줄 코드로 공식 입력만 읽고 직접 학습·예측한다. 외부 제출값, 타깃 기반 특성,
  고정 예측 덮어쓰기 경로를 관찰하지 않았다. CV가 없어 일반화 성능은 확인할 수 없다.
- 최종 특성을 비용·주요 범주 중심으로 축소한다. 비용 결측을 0 지출로 해석하는 부분은
  우리 결측 의미 규칙에 맞게 고쳐야 한다. imputer/OHE는 학습 fold에 fit한다.
- depth 5, 약 730 trees, learning rate 약 0.0667, L1/L2·subsampling은 작은 사전 지정
  비교의 출발점으로만 사용한다. 외부에서 정한 특성 목록을 보편적인 최적값으로 취급하지 않는다.
- 페이지 관측: V3/8, Best 0.81599 V2. 내려받은 최신 소스와 점수 버전의 일치는 미확인.

### Dmitry Shekhov: CatBoost + 임계값

- 공식 입력에서 직접 CatBoost 확률을 만든다. 다른 제출 CSV를 읽지 않는다.
  depth 4, learning rate 0.015, 최대 1,000 iterations를 사용한다(388줄).
- ROC의 TPR-FPR를 최대화한 임계값을 holdout에서 선택하고 같은 holdout에서 평가한다(444줄).
  이는 accuracy를 직접 최대화하는 기준도 아니며, 선택 후 평가가 낙관적일 수 있다.
- 우리 적용은 0.5를 대조군으로 유지하고 내부 그룹 검증에서 임계값을 결정한다.
  페이지 관측: V41/41, Public/Best 0.81318 V41.

### Misael Ribeiro: CatBoost + backward selection

- 공식 입력으로 객실 분해·총지출을 만들고 CatBoost의 SequentialFeatureSelector를 사용한다(155줄).
  다른 제출값 또는 테스트 정답을 읽는 경로를 관찰하지 않았다.
- 검증 전에 전체 train에 imputer를 fit하며 test에도 별도로 fit한다(108줄).
  검증 전처리와 train/test 전처리 불일치를 수정해야 한다.
- 내부 CV로 특성을 선택하는 기법 자체는 정상적이다. 최종 선택 성능에는 별도 바깥 SGKF가 필요하다.
  큰 backward 탐색 대신 소수 제거 후보부터 시작한다.
- 페이지 관측: V9/9, Best 0.81271 V8. 최고 점수 버전의 소스 일치는 미확인.

### Samuel Cortinhas: 설명이 좋은 FE·평균 앙상블

- 공식 특성에서 그룹·성씨·객실 관계를 구성하고 타깃을 분리해 학습한다.
  CatBoost와 LightGBM의 10-fold test 확률을 평균한다(905줄 이후).
- 타깃 없는 그룹·객실 보완은 참고할 수 있으나 학습형 imputer/scaler는 fold 안에서 fit하고
  그룹을 분리해 평가한다. 이미 v2에서 실패한 광범위 특성 추가를 그대로 반복하지 않는다.
- 최종 임계값은 테스트 양성 비율 0.519 가정에 의존한다(979줄).
  이 비율이 독립 검증에서 정해졌는지 확인할 수 없어 우리 적용에서는 제외한다.
- 페이지 관측: V47/47, Public/Best 0.80874 V47. 비슷한 제목의 다른 작성자 노트북과 구별했다.

## 제외·보류한 후보

| 후보 | 관찰한 이유와 판정 |
| --- | --- |
| [bsthere · 0.82137 Solution](https://www.kaggle.com/code/bsthere/spaceship-titanic-0-82137-solution) | `use_best_public_override=True`로 테스트 행수에 맞는 내장 비트열이 모델 예측을 덮어씀. **고득점 근거 제외**. 값의 출처는 불명이며 실제 정답이라고 단정하지 않음. 자동 제출 기본값도 켜져 있음 |
| [junaid512 · Spaceship Titanic](https://www.kaggle.com/code/junaid512/spaceship-titanic) | 전체 train 라벨로 만든 LOO 그룹 특성을 CV 전에 입력(274–333줄). 검증 행 라벨이 학습 행 특성에 들어감. stacker 자체 학습값 평가·전체 OOF 임계값 탐색·전역 pseudo-label도 사용. **검증 누출로 제외** |
| [arunklenin · Advanced FE](https://www.kaggle.com/code/arunklenin/space-titanic-eda-advanced-feature-engineering) | 직접 학습하는 부분도 있으나 최종 제출은 다른 3개 제출 CSV와 자체 예측의 OR(2061줄 이후). 표시 Best 0.82066 V73. **표시 점수 구현 제외**, 자체 FE는 별도 재구현 가능 |
| [ravi20076 · Models V2](https://www.kaggle.com/code/ravi20076/spaceshiptitanic-models-v2) | 파생 parquet를 읽고 최종 예측을 다른 2개 제출 CSV와 OR(1007줄 이후). **제외**. 파생 데이터 생성 경로도 추가 확인 필요 |
| [abdmental01 · Advance Feature.E](https://www.kaggle.com/code/abdmental01/spaceship-advance-feature-e) | 학습 블록 뒤 기존 제출 CSV들을 읽어 OR. TF-IDF/SVD를 train/test에 별도로 fit하는 문제도 있음. **최종 점수 구현 제외** |
| [jbomitchell · Exponentially weighted ensemble](https://www.kaggle.com/code/jbomitchell/exponentially-weighted-ensemble-spaceship-titanic) | 외부 데이터셋의 기존 submission 파일들을 읽어 앙상블(27줄 이후). **프로젝트 규칙으로 제외**. 기초 모델 생성·OOF 검증 없이 점수 기반 가중치를 복사하지 않음 |
| [gyogyocat · Baseline GBM](https://www.kaggle.com/code/gyogyocat/baseline-gbm) | R LightGBM 학습 뒤 외부 `/kaggle/input/blend1/submission.csv`를 읽음(374줄). 표시 Best 0.81903 V20. **최종 점수 구현 제외** |
| [r2303254 · ImprovedBaselineV2](https://www.kaggle.com/code/r2303254/inf2008-p5-2-improvedbaselinev2) | 파생 parquet와 다른 제출 CSV를 최종 결과에 사용(1987줄 이후). **최종 점수 구현 제외** |
| [jimliu · Modularity FE](https://www.kaggle.com/code/jimliu/0-81669-misaelcribeiro-solution-modularity-fe) | 외부 Headjack 허브의 원격 변환. 소스만으로 변환의 학습 데이터·타깃 독립성을 검증할 수 없음. **보류**. 제목의 0.81669를 검증된 정상 점수로 확정하지 않음 |
| [hseyinkalkan · XGBClassifier+Optuna](https://www.kaggle.com/code/hseyinkalkan/spaceshiptitanic-xgbclassifier-optuna) | 전체 X,y에 학습한 LGBM을 X_val,y_val에서 permutation importance 평가하고 특성 선택한 뒤 같은 데이터로 CV. 표시 Best 0.81809 V20. **엄격한 검증 참고 목록에서 제외**. 테스트 정답 치팅 증거와 구별 |
| [cv13j0 · NN + Feature Eng](https://www.kaggle.com/code/cv13j0/spaceship-titanic-nn-model-feature-eng) | 전체 y로 만든 supervised KNN 특성을 별도 바깥 NN CV 전에 저장(546줄). 최종 특성에 KNN_K1_01 사용. 내부 KNN OOF와 바깥 fold 분리가 확인되지 않아 **보류**. 전역 TransportedPercentage는 최종 특성에서 제외됨 |

[umanglodaya의 XGBoost](https://www.kaggle.com/code/umanglodaya/spaceship-titanic-xgboost)는 공식 데이터 직접 학습 계열이지만
Viktor와 코드·기법이 매우 유사해 독립 추천 수에 넣지 않았다. 점수는 추정하지 않았다.
검색에서 보인 meenalsinha 등 추가 후보는 18개 소스 점검 수에 포함하지 않는다.

## 우리 프로젝트의 다음 가설

1. 기존 계획의 작은 CatBoost depth/L2 비교를 유지한다. 챔피언은 baseline-001, OOF 0.818820 / LB 0.80780이다.
2. twinpilgrim/Viktor를 참고해 **기본·이름 수정 특성만 사용하는 작은 XGBoost**를 비교한다.
   기존 full-v2 XGBoost 0.817784와 구별되는 가설이다. 외부 특성 목록을 그대로 답처럼 사용하지 않는다.
3. 특성 제거를 작은 사전 지정 ablation으로 평가한다. 선택을 학습하면 nested SGKF 안에서 수행한다.
4. 확률 보정·임계값·앙상블은 내부 그룹 검증이 있는 경우만 진행한다.
   기존 v2의 log loss 개선이 accuracy 개선으로 이어지지 않았다는 결과를 유지한다.
5. 그룹·객실 보완은 이미 v2에서 전체 개선을 주지 않았으므로 통째로 반복하지 않는다.
   충돌 처리·특성 축소처럼 구체적인 차이가 있는 경우에만 분리해 검증한다.

이번 작업은 조사·소스 점검·기록만 수행했다. 신규 학습, Kaggle 제출, 계정 변경은 없다.
다음 실행 전 공식 데이터·그룹 구조를 확인하는 명령:

```powershell
uv run python scripts/audit_data.py
```
