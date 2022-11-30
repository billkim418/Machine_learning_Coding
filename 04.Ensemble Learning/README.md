# Chapter 3 : Anomaly detection

이번 튜토리얼에서는 이상치 탐지 알고리즘을 밀도(Density-based)기반, 모델(Model-based)기반으로 설명하고 해당 모델들이 어떻게 실제 데이터셋에 적용되는지 살펴보겠습니다. 추가적으로 해당 글은 [고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.

---
## Chapter
-  해당 튜토리얼에서는 아래와 같은 순서대로 진행됩니다.
1. 이상치 데이터(Anomlay data)란?
2. Model-based Anomaly detection(Auto encoder)
3. 국제 유가 데이터셋 적용
4. 
5. Feedback
6. Reference
## 이상치 데이터(Anomaly data)란?
### 이상치 데이터?
- Novel data- Positive
- Anomaly, Foregery, Out-of-distribution data- Negative <br>
- 크게 위처럼 2가지의 의미로 불려지고는 하는데, 긍정적 의미로 이상치 탐지 모델을 만드는데 도움이 되는 Novel(새로운,신기한) data라 불린다. 또한 부정적 의미로는 Anomaly(변칙,이례), Forgery(위조의) data 등의 나쁜 의미를 가진 데이터로 불리기도 합니다. 하지만 이상치 데이터란 실제 현실에서 매우 소중한 데이터이고 이를 잘 활용하는것이 무엇보다 중요하다 할 수 있습니다.

### 이상치 데이터의 정의
>“Observation that deviate so much from other observation as to arouse suspicions that they were <span style='color:yellow'> generated by a different mechanism</span> Hawkins, 1980”<br>
- 해당 인용구를 통해 이상치 탐지는 다른 메커니즘에 의해 생성된 데이터를 지칭하는 것을 알 수 있습니다.
>“Instance that their true probability density is very law (Harmeling et al, 2006)”
- 해당 인용구에서는 이상치를 실제 데이터 분포가 낮은 데이터를 지칭하는 것을 알 수 있습니다.

### Anomaly data vs noise data
여기까지 이상치 데이터의 정의에 대해 알아 봤습니다. 하지만 이렇게 정의만 들으면 이상치 데이터와 노이즈 데이터가 상이한것을 알 수 없습니다. 따라서 아래 그림과 설명을 통해 이상ㅊ티 데이터와 노이즈 데이터를 구분해 보겠습니다.<br>
우선 노이즈 데이터의 특징부터 살펴보겠습니다.
- 노이즈는 측정 과정에서의 무작위성(randomness)에 기반함
- 노이즈 데이터는 엄밀히 말하면 모든 데이터에 존재허고 이는 삭제가 불가능함
다음으로는 실제 노이즈 데이터와 이상치 데이터 어떻게 다른지 그림을 통해 살펴보겠습니다.
![image](https://user-images.githubusercontent.com/68594529/201829848-e9a9d60e-956f-4242-959f-e5437140ba94.png)
- (a) outlier data : 특정 데이터(A)가 명확히 다른 데이터들과 떨어져 분포하고 있음
- (b) noise data : 특정 데이터(A)가 명확히 다른 데이터들과 떨어져 분포하고 있다고 말 할 수 없음
이처럼 이상치 데이터와 노이즈 데이터는 명확히 다르고 우리는 이 중 이상치 데이터를 탐지하는 알고리즘에 대해 알아보겠습니다.
--- 
아래와 같이 분포되어 있는 데이터셋이 있을 때, A,B라는 이상치 데이터를 우리는 어떻게 탐지할수 있을지 생각해 보겠습니다.
![image](https://user-images.githubusercontent.com/68594529/201831043-c53aff20-79ca-4828-b7ef-0c3989f37a34.png)<br>
#### Q : 여기서 한가지 의문점이 생기실겁니다. A,B라는 이상치 데이터를 비교적 잘 알려진 분류 알고리즘을 이용해 탐지할 순 없을까요?<br>
#### A : 정답은 힘들다입니다. 그 이유는 분류 알고리즘으로는 소수(Miniority)의 클래스 구분이 어렵기 때문입니다.

![image](https://user-images.githubusercontent.com/68594529/201832124-d04f9f4a-e0f5-406d-a09e-e9580c6e521b.png)

위의 그림을 보면 좀 더 쉽게 이해가 가리라 생각됩니다. 우선 왼쪽의 분류 모델같은 경우 A,B를 탐지 못하는 반면에 오른쪽의 이상치 탐지 모델은 정상인 데이터들을 학습함으로써 정상이 아닌 즉 이상치인 A, B 데이터를 탐지해 내게 됩니다.

---
이제부터 본격적인 이상치 탐지 방법론들에 대해 알아보겠습니다.
### Model-based Anomaly detection
- 이상치 탐지 방법론에는 크게 밀도기반의 density-based anomaly detecton과 모델기반의 model-based anomaly detection으로 나뉩니다. 그 중에서 이번 튜토리얼은 모델 기반의 방법론들에 대해 소개하고 이를 time-sires dataset에 적용시켜보는 시간을 가져보겠습니다.

#### Auto-Encoder
![image](https://user-images.githubusercontent.com/68594529/202369773-815347f4-9a12-4126-b3b7-16d24f9f81b4.png)

- 오토인코더란 위의 그림처럼 입력과 출력이 동일한 인공 신경망 구조입니다.
- 입력층과 출력층의 차원은 동일하지만 은늑칭은 입력층의 차원 수를 넘을 수 없습니다.<br>
그렇다면 같은 출력을 가지는 모델을 학습함으로써 얻는 이득은 무엇일까요?<br>
1. 차원 축소의 목적으로 AE를 학습시킨 후에 추후 잠재벡터인 feautre를 이용하여 latent한 특징 포착
2. 입력정보와 AE출력 정보간 차이를 이용한 분석을 통해 이상치를 분석함

AE의 장점
1. 중요 Feature만을 압축하기에 용량도 작고 품질도 더 좋음
2. 차원의 저주를 예방할수 있음
3. 복원이 잘도지 않을 경우, 어느 위치에서 차이가 크게 나는지를 쉽게 알 수 있음

AE의 단점
- 입력에 대한 약간의 변형에도 모델이 민감하게 반응함
- 따라서 단점을 보완하기 위해 noise를 첨가해 nosie가 제거된 결과값이 나오도록 함, 이 과정은 모델을 robust하게 함
- Noise는 보통 Random Gaussian Nosie를 사용함

### 국제 유가 데이터 이상치 탐지

### DATASET
- 출처 : [KREI : 해외곡물시장정보](http://www.krei.re.kr:18181/new_sub14)에서 가져온 국제 유가 데이터를 사용
- 데이터셋 구성 : 2012-11-16 ~ 2022-11-15까지의 데이터를 사용
- INFO : 2550 Rows, 3 feature(WTi, Brent, Dubai)
- WTi: 서부텍사스원유, 전 세계 원유 시장에서 가장 영향력이 강한 원유로서 품질이 가장 좋음
- Brent : 가장 광범위한 지역으로 수출되고 있는 원유, 바다에서 추출되며 유통에 강점이 있음, 영국 북해등지에서 생산
- Dubai : 중동산 대표 원유, 중동 아랍지역에서 생산되는 원유로서 선물 거래가 아닌 장기 공급 계약형태로 거래된다는 차이를 보임
- 이번 튜토리얼에서는 해당 데이터셋을 통해 원유가 거래 데이터셋의 이상치를 탐지해보고 시각화까지 적용해보겠습니다.
--- 

- 우선 전체적인 3가지 유가 데이터의 분포 그래프를 살펴보겠습니다.
![image](https://user-images.githubusercontent.com/68594529/202599784-d4196fbe-c8c3-4995-a1a1-0ff4dfe1a264.png)<br>
해당 그래프를 통해 기간마다 조금씩 다르지만 전체적으로 WTi가격이 낮음을 확인할 수 있습니다.

- 다음으로는 위에서 배운 오토인코더를 LSTM을 이용하여 구현해보겠습니다.
```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler


# seed 설정
tf.random.set_seed(418)
np.random.seed(418)

# Hyper parameter 설정
window_size = 10
batch_size = 32
features = ['WTi', 'Brent', 'Dubai']
n_features = len(features)
TRAIN_SIZE = int(len(oil)*0.7)

# Scaling
scaler = StandardScaler()
scaler = scaler.fit(oil.loc[:TRAIN_SIZE,features].values)
scaled = scaler.transform(oil[features].values)

# keras TimeseriesGenerator를 이용해 데이터셋 만들기
train_gen = TimeseriesGenerator(
    data = scaled,
    targets = scaled,
    length = window_size,
    stride=1,
    sampling_rate=1,
    batch_size= batch_size,
    shuffle=False,
    start_index=0,
    end_index=None,
)

valid_gen = TimeseriesGenerator(
    data = scaled,
    targets = scaled,
    length = window_size,
    stride=1,
    sampling_rate=1,
    batch_size=batch_size,
    shuffle=False,
    start_index=TRAIN_SIZE,
    end_index=None,
)

print(train_gen[0][0].shape)  # (32, 10, 5)
print(train_gen[0][1].shape)  # (32, 5)


# Modleing
# 2개 층의 LSTM으로 인코더 만듬
# RepeatVector는 input을 window_size만큼 복사해줌
model = Sequential([
    # >> 인코더 시작
    LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, n_features)),
    LSTM(16, activation='relu', return_sequences=False),
    ## << 인코더 끝
    ## >> Bottleneck
    RepeatVector(window_size),
    ## << Bottleneck
    ## >> 디코더 시작
    LSTM(16, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=False),
    Dense(n_features)
    ## << 디코더 끝
])


# Checkpoint
# 학습을 진행하며 validation 결과가 가장 좋은 모델을 저장해둠
checkpoint_path = 'C:\\Users\\kimhongbum\PycharmProjects\BA\\mymodel.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

# Earlystopping
# 학습을 진행하며 validation 결과가 나빠지면 스톱. patience=5로 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='mae', optimizer='adam',metrics=["mae"])


hist = model.fit(train_gen,
          validation_data=valid_gen,
          steps_per_epoch=len(train_gen),
          validation_steps=len(valid_gen),
          epochs=50,
          callbacks=[checkpoint, early_stop])


model.load_weights(checkpoint_path)
# <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fa7e4312910>
```


```python
import copy

test_df = copy.deepcopy(oil.loc[window_size:]).reset_index(drop=True)
test_df['Loss'] = mae_loss

threshold = 0.5
test_df.loc[test_df.Loss>threshold]

```

- 아래는 해당 모델의 학습 결과 그래프입니다.
![image](https://user-images.githubusercontent.com/68594529/202600021-f725d892-e25f-4021-b6fb-d6b71f6c9afa.png)<br>

- 해당 코드를 통해 아래 인덱스가 이상치임을 탐지하였습니다.<br>
- [94, 96, 97, 98, 146, 158, 159, 164, 165, 167, 633, 667, 668] 총 13개의 이상치가 탐지되었습니다.
![image](https://user-images.githubusercontent.com/68594529/202600482-5c69b331-0a2d-469d-b79a-20bbc0b4767c.png)<br>

- 다음으로는 해당 이상치 데이터들을 그래프릍 통해 살펴보겠습니다.

```python
fig = plt.figure(figsize=(12,15))

# 가격들 그래프입니다
ax = fig.add_subplot(311)
ax.set_title('Open/Close')
plt.plot(test_df.Date, test_df.WTi, linewidth=0.5, alpha=0.75, label='Close')
plt.plot(test_df.Date, test_df.Brent, linewidth=0.5, alpha=0.75, label='Open')
plt.plot(test_df.Date, test_df.Dubai,        'or', markevery=[mae_loss>threshold])

# 오차율 그래프입니다
ax = fig.add_subplot(313)
ax.set_title('Loss')
plt.plot(test_df.Date, test_df.Loss, linewidth=0.5, alpha=0.75, label='Loss')
#plt.plot(test_df.Date, test_df.Loss, 'or', markevery=[mae_loss>threshold])
```
![image](https://user-images.githubusercontent.com/68594529/202600686-bc04b9d8-b675-4d20-9372-9d38b8b4aac2.png)

- 이처럼 오토인코더를 통하여 특정 지점에서의 이상치를 탐지할수 있었습니다.
- 또한 loss 값 그래프를 통해 보면 2020년, 2022년 그 변동폭이 심함을 확인할수 있습니다. -> 또한 실제로 이러한 변동 폭이 발생하는 지점에서 이상치들이 탐지되었습니다
- 만약 thershold 값을 좀더 robust하게 준다면 더 많은 이상치를 탐지할수 있겠지만 현실세게에서 실제로 그러한 지점을 이상치라 지정하기에는 많은 도메인 지식이 필요하다 생각합니다.

## Conclusion
[그래프로 본 2020년 국제유가…“1년 사이 2배 올랐다” - 에너지 신문](https://www.energy-news.co.kr/news/articleView.html?idxno=75546)<br>
[국제유가 일주일 새 또 급락…주유소 반영은 “3주 이내” - 한겨례 신문](https://www.hani.co.kr/arti/economy/marketing/1050842.html)

- 해당 기사들을 보면 최근들어 특정 년도인 2020년, 2022년도에 유가가 상승 혹은 하락했음을 확인할 수 있습니다.
- 이러한 트렌드들은 시계열 데이터들에서 이상치로 탐지 될 수 있으며 이를 AE는 잘 탐지해냈다고 해석하였습니다.
- 이상치 탐지 알고리즘이 정답은 아니겠지만 해당 알고리즘을 통하여 우리는 시계열 데이터에서 다른 패턴을 보이는 데이터들을 파악할수 있습니다.

## Feedback

- 해당 튜토리얼에서 이상치 탐지 설명을 시계열 데이터로 설명했다면 직관적으로 이해가 빨랐을것 같습니다.
- 좀 더 다양한 시계열 이상치 탐지 알고리즘의 성능을 비교했다면 좋았을 것 같습니다. 예를 들어 고전적인 시계열 분해, ARIMA 등

---
## References
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)<br>
[Intorduction - pape](https://www.researchgate.net/publication/324532542_Smart_Driving_Behavior_Analysis_Based_on_Online_Outlier_Detection_Insights_from_a_Controlled_Case_Study)<br>
[국제 유가 데이터](http://www.krei.re.kr:18181/new_sub14)
