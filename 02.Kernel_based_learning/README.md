## Chapter 2 : Kernel-based Learning

이번 튜토리얼에서는 서포트 벡터 머신의 원리를 다루고 더 나아가 비슷한 선형 분류기인 LDA와의 비교를 진행해보겠습니다. 우선적으로 해당 글은 기본적으로 
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.

---
### Chapter
- 해당 튜토리얼에서는 아래와 같은 순서대로 진행됩니다.
1. SVM 이론적 원리 및 구현
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

Q : 처음에 SVM이 선형 분류기라는 얘기를 하셨는데 그렇다면 비선형적인 데이터 어떻게 해결할수 있을까요?<br>
A : 원래 공간이 아닌 선형 분류가 가능한 더 고차원의 공간으로 데이터를 보내서(mappin) 후에 이를 학습하면 됩니다. 해당 과정에서 커널함수가 작용을 합니다.

여기서 커널 트릭(Kernel trick) 함수란 저차원의 데이터를 고차원의 공간에 매핑시켜 주는 함수를 의미합니다. 이 때, 고차원에서 데이터는 항상 두 벡터간의 내적으로만 존재하므로 이러한 커널 트릭 함수의 종류는 다양하게 사용될 수 있습니다.

