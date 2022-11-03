## Chapter 2 : Kernel-based Learning

이번 튜토리얼에서는 서포트 벡터 머신의 원리를 다루고 더 나아가 비슷한 선형 분류기인 LDA와의 비교를 진행해보겠습니다. 우선적으로 해당 글은 기본적으로 
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.

---
### Chapter
- 해당 튜토리얼에서는 아래와 같은 순서대로 진행됩니다.
1. SVM 이론적 원리 
2. SVM에서의 Kernel 작용
3. Kernel Fisher Dsicriminant Anlysis

### SVM의 이론적 원리 및 구현
- 우선적으로 우리는 SVM이 선형 분류기라는 것을 알아야 합니다. 그렇다면 아래와 같은 그림이 주어져 있을 때 어떤 방식으로 분류해야 할까요? 
![02_1_Kernel-based Learning_SVM](https://user-images.githubusercontent.com/68594529/199439258-9627e91a-51bf-4a27-a532-c37586c78e40.png)

Q : B의 그림처럼 나누는것이 정말 좋은 방법이라는 근거가 있을까요?<br>
A : 이러한 근거를 마련하기 위해서 VC_dimension이라는 개념을 알아야 합니다. VC dimension이란 특정 함수가 얼마나 복잡한지 즉 Capacity를 측정하는 지표입니다. 즉 함수 H에 의해 최대로 shatter 가능한 점의 숫자가 곧 VC dimension이라고 할 수 있습니다.

- 여기서 shatter의 의미를 설명하기 전에 Dichotomy란 개념을 추가적으로 설명하겠습니다.간단히 Dichotomy란 특정 집합이 있다면 이를 이분법적으로 나눈다는 개념입니다. 예를들어 A, B ,C 라는 점이 존재한다면 이는 총 {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C} 총 8가지의 구분되는 경우가 존재합니다. 이처럼 특정 조건을 만족하는 set으로 나누는 것을 Dichotomy라고 표현합니다. 그렇다면 shatter하다는 것은 무슨 의미일까요? 바로 특정 함수가 Dichotomy를 모두 표현할수 있느냐입니다. 가능하다면 shatter하다고 표현하는 것입니다.

- 여기까지 VC dimension의 개념에 대해 알아봤는데요. 그렇다면 SVM은 어떤 식으로 선을 분리하는 것일까요? 정답은 아래 그림처럼 margin을 최대화 하는 식으로 분리하는 것입니다.

![image](https://user-images.githubusercontent.com/68594529/199475366-bb87c54f-0f2c-4336-85d7-66217327d190.png)

- maring이란 해당 분류경계면으로부터 가장 가까운 점들과의 거리로 정의되고 오른쪽 그림은 이를 법선벡터를 이용해 표현한 그림입니다.

- Q : 여기서 한가지 의문점이 생기는데 과연 마진이 최대화가 되면 VC dimension이 최소화가 될까요?
- A : 구조적 위험 최소화(Structural Risk Minimization) 접근법을 통해 해결 가능합니다.

 - 구조적 위험 최소화의 수식은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/68594529/199477397-71e4d44e-e204-4e98-8562-80af5b700869.png)

- VC dimension의 수식의 uppber bound는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/68594529/199479054-09142359-a748-44d8-ac12-bc6aaa069deb.png)

해당 2가지 수식을 조합하면 아래와 같은 결론을 얻을 수 있습니다. 우선적으로 
$∆^2 ↑ -> ⌈\frac{R^2}{∆^2} ⌉ ↓ -> min( ⌈\frac{R^2}{∆^2} ⌉,D) ↓ -> h ↓ ->  B ↓ -> R[f] ↓ $

따라서 마진이 최대화 되면 VC dimension이 최소화되고 이는 즉 Capacity 항이 최소화되게 됩니다.

여기까지 이해하셨다면 이제 본격적으로 SVM을 모델링해 보겠습니다.<br>
SVM Modeling process
1. 마진이 최대화 되는 목적 함수 설정 $min 1/2 ‖𝑤‖^2 +\sum_{i=1}^ n ξ_𝑖$
2. 모든 데이터에 대하여 제약식 설정s.t  $𝑦_𝑖 (𝑤^𝑇 𝑥_𝑖+𝑏)≥1−ξ_𝑖,  ξ_𝑖≥0,  ∀𝑖$
3. 최적화를 통한 문제 해결 : 라그랑주, KKT conditions 등을 이용한다 -> 자세한 내용은 최적화 내용이므로 생략하고 진행하겠습니다.

해당 전개 과정을 보면 갑자기 ξ_𝑖가 나타났습니다. 해당 기호는 크사이(Penalty)로서 모든 데이터에는 노이즈(noise)가 존재하고 이를 반영하기 위한 장치입니다!

<img src="https://user-images.githubusercontent.com/68594529/199502831-9c1e0b1a-2b0f-4561-a366-a14aaeba1207.png" width="600" height="400"/>

위의 그림처럼 우리는 패널티를 줌으로써 좀더 soft하게 SVM을 설계할 수 있습니다. 이처럼 SVM에는 크사이 이외에도 Performance를 위한 장치들이 더 있습니다.<br>
예를 들면, 오분류 비용 C 즉 패널티를 줄이면서 마진까지 같이 줄일 것인지 혹은 패널티를 받더라도 마진을 넓게 잡도록 학습시킬것인지 조절 가능합니다

### SVM에서의 Kernel 작용

Q : 처음에 SVM이 선형 분류기라는 얘기를 하셨는데 그렇다면 비선형적인 데이터는 어떻게 해결할수 있을까요?<br>
A : 원래 공간이 아닌 선형 분류가 가능한 더 고차원의 공간으로 데이터를 보내서(mappin) 후에 이를 학습하면 됩니다. 해당 과정에서 커널함수가 작용을 합니다.

여기서 커널 트릭(Kernel trick) 함수란 저차원의 데이터를 고차원의 공간에 매핑시켜 주는 함수를 의미합니다. 이 때, 고차원에서 데이터는 항상 두 벡터간의 내적으로만 존재하므로 이러한 커널 트릭 함수의 종류는 다양하게 사용될 수 있습니다.

커널 트릭 함수는 또한 단지 두 벡터간의 내적을 계산할수 있어야할 뿐만 아니라 아래의 Mercer's Theorem을 만족해야합니다. 해당 이론은 아래 그림을 참고해주세요.
![image](https://user-images.githubusercontent.com/68594529/199635742-b840bfeb-ddfe-4901-b31e-88d1d7ab603c.png)<br>
출처 : https://sonsnotation.blogspot.com/2020/11/11-1-kernel.html
해당 정리를 요약하면 아래와 같습니다.
-> Kernel 함수 K 가 실수 scalar 를 출력하는 continuous function일 것 <br>
-> Kernel 함수값으로 만든 행렬이 Symmetric(대칭행렬)이다.<br>
-> Positive semi-definite(대각원소>0)라면 $K(xi, xj) = K(xj, xi) = <Φ(xi), Φ(xj)>$를 만족하는 mapping Φ 가 존재한다. 즉, Reproducing kernel Hilbert space라는 의미입니다.

위와 같은 정리를 만족하는 대표적인 kernerl 트릭 함수의 종류는 아래와 같습니다.
- Polynomial : $K(x,y) = ( x \cdot y + c) ^d
- Linear : $K(x,y) = (x \cdot y^T)
- Gaussian(RBF) : $exp(-\frac {||x-y||^2} {2\sigma^2})$

#### Python code
```python
import numpy as np

class SVM:
  #kernel 함수
  def __init__(self, kernel='linear', C=10000.0, max_iter=100000, degree=3, gamma=1):
    self.kernel = {'poly'  : lambda x,y: np.dot(x, y.T)**degree,
                   'rbf'   : lambda x,y: np.exp(-gamma*np.sum((y - x[:,np.newaxis])**2, axis=-1)),
                   'linear': lambda x,y: np.dot(x, y.T)}[kernel]
    #오분류 비용 C
    self.C = C
    #반복 시행 횟수
    self.max_iter = max_iter
  # np.clip(array, min, max)함수를 이용하여 square로 변환함
  # 최적의 선을 찾기 위한 반복 수행 과정중 square 재구축 과정(min,max 벗어나는 값 재구축됨)
  def restrict_to_square(self, t, v0, u):
    t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
    return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]
  # Optimization
  def fit(self, X, y):
    self.X = X.copy()
    self.y = y * 2 - 1
    self.lambdas = np.zeros_like(self.y, dtype=float)
    self.K = self.kernel(self.X, self.X) * self.y[:,np.newaxis] * self.y
    
    #반복 수행하며 최적의 분류 경계면을 구함
    for _ in range(self.max_iter):
      for idxM in range(len(self.lambdas)):
        idxL = np.random.randint(0, len(self.lambdas))
        Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
        v0 = self.lambdas[[idxM, idxL]]
        k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
        u = np.array([-self.y[idxL], self.y[idxM]])
        t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
        self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
    
    idx, = np.nonzero(self.lambdas > 1E-15)
    self.b = np.mean((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx])
  
  #최종 분류면 
  def decision_function(self, X):
    return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b
  
  #예측 시행
  def predict(self, X):
    return (np.sign(self.decision_function(X)) + 1) // 2
```
다음으로는 Sklearn의 wrapper 모델인 SVC와 성능 비교를 진행해보겠습니다.

위의 파이썬 코드를 통해 SVM 분류기를 만들어보았습니다. 그렇다면 과연 해당 코드와 실제 Sklearn의 SVC와의 비교를 진행해보겠습니다.
- 우선 분류 경계면을 생성하고 이를 비교하기 위한 test_plot 함수를 생성하겠습니다.
```python
def test_plot(X, y, svm_model, axes, title):
  plt.axes(axes)
  xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
  ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
  xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
  rgb=np.array([[210, 0, 0], [0, 0, 150]])/255.0
  start_time = time.time()
  svm_model.fit(X, y)
  z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  end_time = time.time()
  print("WorkingTime %s time : %s sec" % (svm_model, end_time-start_time))
  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
  plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
  plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
  plt.title(title)
```
위의 함수를 토대로 결정 경계면이 어떻게 형성되는지 차이를 보고 추가적으로 함수 알고리즘의 시간적 차이를 살펴보겠습니다.
데이터셋 예시 : 원형 데이터, 선형 데이터, 비선형 데이터 
```python
import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs, make_circles
from matplotlib.colors import ListedColormap

X, y = make_circles(100, factor=.1, noise=.1)
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
test_plot(X, y, SVM(kernel='rbf', C=10, max_iter=60, gamma=1), axs[0], 'OUR ALGORITHM')
test_plot(X, y, SVC(kernel='rbf', C=10, gamma=1), axs[1], 'sklearn.svm.SVC')

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1.4)
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
test_plot(X, y, SVM(kernel='linear', C=10, max_iter=60), axs[0], 'Our Algorithm')
test_plot(X, y, SVC(kernel='linear', C=10), axs[1], 'sklearn.svm.SVC')

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
test_plot(X, y, SVM(kernel='poly', C=5, max_iter=60, degree=3), axs[0], 'Our Algorithm')
test_plot(X, y, SVC(kernel='poly', C=5, degree=3), axs[1], 'sklearn.svm.SVC')
```
||Sklearn svm|our svm|
|:---:|:---:|:---:|
|원형 데이터|0.285 sec|1.761 sec|
|선형 데이터|0.185 sec|0.289 sec|
|비선형 데이터|0.177 sec|0.743 sec|

- 원형 데이터셋 result
![image](https://user-images.githubusercontent.com/68594529/199649349-e4d3b6b6-bdf7-412f-8804-433227c48267.png)
- 선형 데이터셋
![image](https://user-images.githubusercontent.com/68594529/199649471-91c760d1-c0f4-46cf-8a7c-c157ec49c7d6.png)
- 비선형 데이터셋
![image](https://user-images.githubusercontent.com/68594529/199649486-24affd05-20f8-4e18-84c7-a532170444e0.png)

해당 결과를 보면 sklean의 svc의 성능이 최소 2배에서 5배까지 차이가 남을 확인할수 있었습니다. 해당 이유는 Sklearn 의 defalut 설정 때문인것 같습니다.
- class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
- skleanr의 default parameter 설정을 보면 shrinking이란 설정이 defalut로 True가 되어 있는 것을 볼 수 있습니다. sklearn에서 참고 한 해당 논문을 보면(https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf) , itertation이 커질수록 shrinking이 커질수록 training time이 줄어든다 되어 있습니다. 하지만 이러한 부분이 저의 논문에서는 구현이 되어 있지 않기 때문에 성능 차이가 발생하였다 판단하였습니다.

---

## Feedback

- 해당 튜토리얼은 SVM 모델 그중에서도 직관적인 classificiation에 초점을 맞춰 진행함으로써 regression 및 anamoly detection에서의 svm의 성능 검증이 부족합니다.
- 다양한 데이터셋에서의 성능검증이 되었다면 사용자들이 상황에 맞게 선택하여 하이퍼 파라미터(C, Gamma)를 선택할 수 있었을것 같습니다.

---
## References
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)<br>
[Sklearn - SVM](https://scikit-learn.org/stable/modules/svm.html#shrinking-svm)<br>
[Shrinking paper] (https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)


