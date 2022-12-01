# Chapter 4 : Ensemble Learning

이번 튜토리얼 에서는 앙상블(Ensemble)이라는 주제를 다뤄 보겠습니다. 앙상블이란 지금까지 튜토리얼에서 다뤄 본 여러 머신 러닝 알고리즘들을 다양한 방식으로 결합 혹은 여러번 학습함으로써 성능을 올리는 방법론입니다. 특히 우리는 그 중에서  현실 데이터셋과 결합해 실제 성능을 비교해보겠습니다.
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.

---
## Chapter
-  해당 튜토리얼에서는 아래와 같은 순서대로 진행됩니다.
1. 앙상블(Ensemble learning)란?
2. 
3. 
5. Feedback
6. Reference

--- 
## 1. 앙상블 학습(Ensemble learning)이란?
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

### XGboost Vanilla 구현
```python


```
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

### 생존 분석을 위한 XGBoost
기존의 XGBoost 모델은 생존 분석을 위해 Cox 및 AFT(Accerlated Failure Time) 두 가지 방법을 제공합니다.
- [Cox 모델](https://forecast.global/insight/underrated-in-ml-intro-to-survival-analysis/) : 시간이나 생존이 아닌 위험만 출력함
- [AFT 모델](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) : 각 샘플의 예상 생존 시간으로 해석되어야 하는 값을 출력함 

해당 튜토리얼에서는 AFT 모델만을 다루겠습니다. Cox 모델에 관심있으신 분들은 위의 링크를 참조하시면 될 것 같습니다. 다음으로 소개해 드릴 AFT 기법은 기본적으로 아래 모형을 가정합니다.
$$lnY = w \bullet x + \sigma Z$$
- Probability Density Function(PDF)
-
|AFT_loss_distribution|Probability Density Function(PDF)|
|:---:|:---:|
|normal|$$\frac{exp(-z^/2)}{\sqrt{2\pi}}$$|
|logsitic|$$\frac{e^z}{(1+e^z)^2}$$|
|extreme|$$e^z e^-expz$$|

## Conclusion

## Feedback

## References
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)<br>
[XGBoost-Cox](https://forecast.global/insight/underrated-in-ml-intro-to-survival-analysis/)



