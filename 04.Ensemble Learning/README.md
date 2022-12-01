# Chapter 4 : Ensemble Learning

이번 튜토리얼 에서는 앙상블(Ensemble)이라는 주제를 다뤄 보겠습니다. 앙상블이란 지금까지 튜토리얼에서 다뤄 본 여러 머신 러닝 알고리즘들을 다양한 방식으로 결합 혹은 여러번 학습함으로써 성능을 올리는 방법론입니다. 특히 우리는 그 중에서 가장 유명한 모델인 XGBoost를 소개하고 이를 생존분석과 연결 이를 실제 데이터셋은 METBRIC(유방암 데이터셋)과의 적용을 해보겠습니다.
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.


--- 
## 앙상블 학습(Ensemble learning)이란?
앙상블이란 간단하게 집단 지성의 아이디어에서 시작되었습니다. 간단하게 말해서 모델을 다양하게 사용하고 결과를 잘 섞어 더 나은 결과를 도출하는 방법론입니다. 해당 방법론을 적용하기 위해 크게 다양성(diversity)과 최종 결과물을 어떻게 결합하는지가 가장 중요합니다. 이를 아래의 앙상블 목적과 결합하여 살펴보겠습니다.<br>
### 앙상블의 목적 : 다수의 모델을 학습하여 오류의 감소를 추구
- 분산의 감소에 의한 오류 감소 : 배깅(Bagging), 랜덤 포레스트(Random Forest)
- 편향의 감소에 의한 오류 감소 : 부스팅(Boosting)
- 분산과 편향의 동시 감소 : Mixture of Experts <br>

![image](https://user-images.githubusercontent.com/68594529/204792606-d266b889-0d1d-4c01-89fc-8fa002e5a048.png)<br>
- 해당 그림을 통해 간단히 배깅과 부스팅의 개념을 살펴볼수 있는데 중심을 잘 맞추지 못한 데이터들을 배깅(Bagging) 같은 경우에는 퍼져 있는 즉 분산이 높은 애들을 중앙에 더 맞추는 개념이고 복잡도가 높은 모델들을 대상으로 한다면 부스팅(Boosting)은 퍼져 있지는 않지만 한쪽에 치우친 데이터들을 타겟에 맞추는 개념이고 배깅과 다르게 단순한 모델들을 주로 사용합니다.


## XGBoost란 무엇인가?
- 해당 모델 같은 경우에는 부스팅을 사용하는 모델로서 GBM 모델의 단점들을 효율적으로 극복하여 향상시킨 모델입니다. 크게 6가지에서 장점을 가지고 있는데 아래 그림 및 설명을 통해 설명 드리겠습니다.
![image](https://user-images.githubusercontent.com/68594529/204954826-19969201-8e0a-4e61-ad34-148650856eaf.png)
1. 병렬화 된 tree 구조 생성(Parallelized tree building)
  - XGBoost는 block라고 부르는 곳에 데이터를 저장
  - 각 블록의 데이터는  compressed column(CSC) format으로 저장되며, 각 칼럼은 해당 변수 값을 정렬한 상태로 저장
  - 다른 알고리즘은 정렬 정보 보존X
  - 해당 부분은 DT에서 가장 시간이 많이 소요되는 부분임
2. 깊이 우선 탐색을 통한 분기점 생성(Tree pruning using depth-first approach) 
  - 좌측부터 우측으로 변수 값이 오름차순으로 정렬되어 있다고 가정
  - 전체 데이터를 균등하게 분할하고 각 분할에 대해 개별적으로 계산하여 최적의 Split을 찾음
3. 효율적인 컴퓨팅을 위한 시스템 디자인(Cache awareness and out-of-core computing)
  - Cache-aware access -> Cache까지 고려하여 저장
  - Out-of-core computing -> 저장 공간을 블록 단위로 쪼개고 오버헤드 방지
  - Block Compression -> 개별 블록을 column으로 압축, 별도의 thread에서 압축 데이터 호출만 진행
  - Block Sharding -> row 기준으로 자른 후 저장
4. 과적합 방지를 위한 정규화 진행(Regularization for avoiding overfitting)
  - 오버피팅을 막기 위하여 사전에 정규화를 진행함
5. 효율적인 결측치 데이터 처리(Efficient handling of missing data)
  - 현실의 많은 데이터는 밀도가 낮거나, 결측치가 많은 경우가 다수 존재함
  - 데이터를 정렬한 후에 분기의 기본방향을 설정(left, right direction) -> 희소패턴 파악이 용이
6. 교차검증을 통한 범용성 증가(In-bult cross-validation capability)
  - Cross-validation을 통한 데이터의 범용 가능성을 증가시킴

## 생존분석이란 무엇인가?
- 생존분석[Survival analysis](https://en.wikipedia.org/wiki/Survival_analysis#:~:text=Survival%20analysis%20is%20a%20branch,and%20failure%20in%20mechanical%20systems.)이란
통계학의 한 분야로, 어떠한 현상이 발생하기까지에 걸리는 시간에 대해 분석하는 방법입니다. 예를 들면, 생명체의 관찰시작부터 사망에 이르는 시간을 분석하는 등이 있습니다.
- 또한 이러한 생존 분석은 예측 유지 관리, 고객 이탈, 신용 위험, 자산 유동성 위험 및 기타 여러 응용 프로그램에 유용합니다.
- 해당 그림은 생존분석 그래프의 예시입니다.
![image](https://user-images.githubusercontent.com/68594529/204943738-b512b9b2-2f44-4603-830d-8a5274470b74.png)
- 그림을 살펴보면 특정 시점을 기준으로 그래프가 급하강 하는데 이러한 생존 곡선을 통해, 우리는 시간에 따른 누적 생존 확률 (이벤트가 발생하지 않음)을 알 수 있습니다.
- 우리는 앞으로 이러한 생존분석 방법론을 XGBoost와 연관지어 보겠습니다. 

## xgbse 모델 소개
- 해당 모델을 만든 연구자들은 아래에서 설명할 통계적 모델링의 2가지 방향성에 초점을 맞췄습니다.
1. 건전한 통계적 특성(보정된 생존 곡선)을 갖지만 표현력과 계산 효율성이 부족한 모델
2. 매우 효율적이고 표현력이 뛰어난 모델이지만 일부 통계적 특성이 부족하다.(위험만 생성하거나 예상 생존시간을 보정하지 않음 -> XGBoost가 여기에 적합)
- 아래는 간단하게 통계적 모델링에 대한 설명을 첨부해 봤습니다.
---
### 통계적 모델링이란?(Statisitical modeling)
다음으로 설명 드릴 개념은 통계적 모델링입니다. 통계적 모델링이란 확률/수학적인 모형을 가지고 현실 세계의 데이터 형성 과정을 모방하는 것입니다. 해당 방법론은 두가지 방향성을 가지고 있습니다.<br>
- 주어진 확률적 데이터 모델에 의해 데이터가 생성된다고 가정합니다. 즉 분포를 가정하는 것이지요.
- 다른 하나는 알고리즘 모델을 사용하고, 데이터 메커니즘은 알 수 없는 것으로 취급(Blackbox)합니다.<br>
---

### 생존 분석을 위한 기존의 XGBoost 
기존의 XGBoost 모델은 생존 분석을 위해 Cox 및 AFT(Accerlated Failure Time) 두 가지 방법을 제공합니다.
- [Cox 모델](https://forecast.global/insight/underrated-in-ml-intro-to-survival-analysis/) : 시간이나 생존이 아닌 위험만 출력함
- [AFT 모델](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) : 각 샘플의 예상 생존 시간으로 해석되어야 하는 값을 출력함 

해당 튜토리얼에서는 AFT 모델만을 다루겠습니다. Cox 모델에 관심있으신 분들은 위의 링크를 참조하시면 될 것 같습니다. 다음으로 소개해 드릴 AFT 기법은 기본적으로 아래 모형을 가정합니다.
$$lnY = w \bullet x + \sigma Z$$
- Probability Density Function(PDF)
|AFT_loss_distribution|Probability Density Function(PDF)|
|:---:|:---:|
|normal|$$\frac{exp(-z^/2)}{\sqrt{2\pi}}$$|
|logsitic|$$\frac{e^z}{(1+e^z)^2}$$|
|extreme|$$e^z e^-expz$$|
### 사용 데이터셋 (METABRIC)
- METABRIC(Molecular Taxonomy of Breast Cancer International Consortium) : 실제 유방암 데이터셋으로서, 유방암의 조기 진단과 함께 유방암의 예후 예측 등 메디컬 분야에서 다양하게 연구 사용되어지는 데이터셋입니다.
- pycox 패키지를 통해 간단하게 호출 및 사용이 가능하다는 장점을 가지고 있습니다.

```python
# importing dataset from pycox package
from pycox.datasets import metabric

df = metabric.read_df()

X = df.drop(['duration', 'event'], axis=1)
y = convert_to_structured(df['duration'], df['event'])

(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2)

## pre selected params for models ##

PARAMS_XGB_AFT = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'aft_loss_distribution_scale': 1.0,
    'tree_method': 'hist', 
    'learning_rate': 5e-2, 
    'max_depth': 8, 
    'booster':'dart',
    'subsample':0.5,
    'min_child_weight': 50,
    'colsample_bynode':0.5
}
# 실험읋 위한 하이퍼 파라미터 설정
PARAMS_XGB_COX = {
    'objective': 'survival:cox',
    'tree_method': 'hist', 
    'learning_rate': 5e-2, 
    'max_depth': 8, 
    'booster':'dart',
    'subsample':0.5,
    'min_child_weight': 50, 
    'colsample_bynode':0.5
}

PARAMS_TREE = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist', 
    'max_depth': 100, 
    'booster':'dart', 
    'subsample': 1.0,
    'min_child_weight': 50, 
}

PARAMS_LR = {
    'C': 1e-3,
    'max_iter': 500
}

N_NEIGHBORS = 50

TIME_BINS = np.arange(15, 315, 15)

```
- 해당 loss 값들의 영향을 또한 C-index로 살펴보겠습니다.
- C-index : 해당 모델의 risk score가 Event즉 생존을 얼마나 잘 반영하는지를 나타내는 지표고 계산 방법은 아래와 같습니다.
- ![image](https://user-images.githubusercontent.com/68594529/205036103-f5adda95-3d6b-40f6-9cf1-4c3917cdf397.png)
```python
# converting to xgboost format
dtrain = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
dval = convert_data_to_xgb_format(X_valid, y_valid, 'survival:aft')

# training model
bst = xgb.train(
    PARAMS_XGB_AFT,
    dtrain,
    num_boost_round=1000,
    early_stopping_rounds=10,
    evals=[(dval, 'val')],
    verbose_eval=0
)

# predicting and evaluating
preds = bst.predict(dval)
cind = concordance_index(y_valid, -preds, risk_strategy='precomputed')
```
- C-index: 0.658
- Average survival time: 163 days
- xgboost 역시 성능 자체는 매우 뛰어남을 알 수 있다.

### 하이퍼 파라미터 변경에 따른 성능 차이 비교

```python
# saving predictions to plot later
preds_dict = {}

# loop to show different scale results
for scale in [1.5, 1.0, 0.5]:
    
    # chaning parameter
    PARAMS_XGB_AFT['aft_loss_distribution_scale'] = scale
    
    # training model
    bst = xgb.train(
        PARAMS_XGB_AFT,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=10,
        evals=[(dval, 'val')],
        verbose_eval=0
    )

    # predicting and evaluating
    preds = bst.predict(dval)
    cind = concordance_index(y_valid, -preds, risk_strategy='precomputed')

    preds_dict[scale] = preds

    print(f"aft_loss_distribution_scale: {scale}")
    print(f"C-index: {cind:.3f}")
    print(f"Average survival time: {preds.mean():.0f} days")
    print("----")
```
|aft_loss_distribution|C-index|Average survival time|
|:---:|:---:|:---:|
|1.5|0.643|201 days|
|1.0|0.651|159 days|
|0.5|0.643|128 days|
- 해당 표의 결과를 볼시 C-index 즉 위험 지표는 굉장히 준수한 편이다. 하지만 생존 기간의 예측이 loss 값 여기서는 noraml 분포인데 분포의 scale에 따라 70일 가까이 상이함을 알 수 있고 이러한 xgboost 생존분석 모델의 단점을 극복할 필요가 있었다. 즉 모델의 신뢰성 부분에 문제가 있었습니다.

## XGBoost 생존분석 모델의 단점 극복(XGBSEDebiasedBCE)
1. 다중 작업 로지스틱 회귀 방법
2. BCE(이진 교차 엔트로피) 접근 방식 차용
- 기존 앙상블의 각 트리에 있는 터미널 노드(leaf)의 입력 데이터를 임베딩(1,0)으로 변환하여 줍니다.(아래 그림 참고)
![image](https://user-images.githubusercontent.com/68594529/205040774-3d353321-61df-4263-87f1-5ee5b72bf4a3.png)

- 해결책 : 간단하게 기본 XGBoost 모델에 의해 생성된 임베딩 위에 로지스틱 회귀를 진행, 이를 각각 다른 사용자 정의에 의해 시간 예측을 진행합니다.

![image](https://user-images.githubusercontent.com/68594529/205041401-9358c0b8-ccc2-44a4-8a24-0eb8cbd39940.png)
- XGBoost가 'Leaf Ocuurence' 임베딩을 통해 Origninal Space로 변경됨
- 각각의 로지스틱 선형회귀가 서로다른 time windows(사용자)에서 Censored 된 데이터를 제거한 뒤 변경된 Features에서 이벤트 확률을 예측하기 위해 사용함
- 편향되지 않는 [KM](https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf) 커브를 생성하는 transforamtion을 통해, 예측된 선형 회귀가 전체 생존 함수를 생성함
- 여기서 KM이란 Kaplan-Meier 공식으로서 로지스틱 회귀등에서 점확률을 추정하는 방식입니다. -> 이를 통해 곡선 생성 가능
---

- XGBKaplanNeighbors : 위의 방식으로 곡선을 생성하는 방법으로써 단점을 극복하였고, 해당 방식으로는 실제 생존 추정치를 구할 수 있습니다. 방식은 아래와 같습니다.
![image](https://user-images.githubusercontent.com/68594529/205047045-d886d918-7ef0-4216-8b74-73f81d6d77ee.png)
- 2번째 과정까지는 동일하게 진행 하지만 3번째 그림에서부터 이웃된 세트안에서 각 샘플에 대해 NN서치가 임베딩된 환경내에서 시행됩니다.
- 이 후에 In Verctorized fashion에서 Neighbor-set을 위해서 KM 추정 방식을 수행합니다.
- 하지만 이 방식은 코스트 계산 값이 매우 높으므로, 간단한 Tree 방식이 제안되었습니다.
---

- XGBKaplanTree : 계산 과정이 복잡한 NN 과정을 생략하는 대신에 XGBoost를 통해 단일 트리를 구현하고 각 노드에서 KM 곡선을 계산하는 간단한 방식입니다.
![image](https://user-images.githubusercontent.com/68594529/205048061-534fd08a-edfb-448d-8f33-58dd43c9cb66.png)
- 기존의 2가지 방식보다 간단하게 3번째 그림에서 NN서치를 하지 않고, 비슷한 군집에 의해서만 묶이는 것을 볼 수 있고 오히려 NN의 편향성을 떨어트리는 결과도 보여줍니다.
- 단, 단일 트리를 피팅하므로 예측력이 나빠질 가능성을 내포하고 있습니다.
## xgbse를 이용한 성능 비교

```python
from xgbse import XGBSEKaplanNeighbors
from xgbse._kaplan_neighbors import DEFAULT_PARAMS
from xgbse.metrics import concordance_index

for scale in [1.5, 1.0, 0.5]:

    DEFAULT_PARAMS['aft_loss_distribution_scale'] = scale

    xgbse_model = XGBSEKaplanNeighbors(DEFAULT_PARAMS, n_neighbors=30)
    xgbse_model.fit(
        X_train, y_train,
        validation_data = (X_valid, y_valid),
        early_stopping_rounds=10,
        time_bins=TIME_BINS
    )

    preds = xgbse_model.predict(X_valid)
    cind = concordance_index(y_valid, preds)
    avg_probs = preds[[30, 90, 150]].mean().values.round(4).tolist()

    print(f"aft_loss_distribution_scale: {scale}")
    print(f"C-index: {cind:.3f}")
    print(f"Average probability of survival at [30, 90, 150] days: {avg_probs}")
    print("----")

```
|aft_loss_distribution|C-index|30 days| 90days|150 days|
|:---:|:---:|:---:|:---:|:---:|
|1.5|0.625|0.9085|0.6855|0.5289|
|1.0|0.646|0.9092|0.6831|0.5287|
|0.5|0.648|0.918|0.6971|0.5321|
- 해당 표를 통해 우리는 모델이 각각의 생존 예측 확률을 일괄적으로 높은 C-index내에서 나타냄을 보여주고 있습니다.
---

## Feedback

## References
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)<br>
[XGBoost-Cox](https://forecast.global/insight/underrated-in-ml-intro-to-survival-analysis/)
[C-index](https://square.github.io/pysurvival/metrics/c_index.html)
[Kaplan-Meier frmulation](https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf) 


