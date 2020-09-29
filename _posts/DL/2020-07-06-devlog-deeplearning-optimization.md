---
layout: post
title:  "[Deep Learning] 딥러닝 학습의 다양한 기술들을 알아보자 [2] Optimization"
subtitle: "Optimization"
categories: devlog
tags: deeplearning
use_math: true
---

이번 장에서는 다양한 최적화 기법을 소개하겠습니다.
<br/>
 <br/>

## 학습 파라메터 초기화

각 층의 가중치(weights)와 편향(bias) 등 학습 파라메터는 초기값 설정이 매우 중요하다고 합니다. 뉴럴네트워크가 풀려는 문제 자체가 non-convex optimization이기 때문에 시작점에 따라 최적지점을 찾지 못하게 될 수도 있습니다.

또한 학습 파라메터의 초기값을 적절하게 설정할 경우 그래디언트 조절에도 유의미한 효과를 낸다고 합니다. 이와 관련해 시그모이드 함수의 1차 미분 그래프를 다시 보도록 하겠습니다.

![](https://i.imgur.com/WpKD6kW.png)

입력값 $x$에 가중치 $w$를 곱하고 편향 $b$를 더한 식을 $t$라고 둡시다. 여기에서 $w$가 100, $b$가 50이라면 $x$가 0.01로 매우 작더라도 $t$는 51이 됩니다. 역전파시 시그모이드 함수를 통과시키면 $σ'(51)$가 반환이 될텐데요, 위 그래프를 보시다시피 $t$가 5만 넘어도 $σ'(t)$는 0에 수렴하기 때문에 그래디언트가 죽어버리는 결과를 초래하게 됩니다. 그래디언트가 지나치게 작아지기 때문에 이후 학습이 사실상 불가능해지는 것이지요. 

이와 별개로 뉴럴네트워크 입력층의 가중치 $W$를 모두 0으로 초기화한다면 어떻게 될까요? 순전파 때는 $W$가 0이기 때문에 두번째 층의 뉴런에 모두 같은 값이 전달됩니다. **미분의 연쇄법칙(chain-rule)**을 떠올려보면 두번째 층의 모든 뉴런에 같은 값이 입력된다는 것은 역전파 때 두번째 층의 가중치가 모두 똑같이 갱신된다는 말이 됩니다. 다시 말해 뉴런이 100개가 됐든 1000개가 됐든 거의 같은 값을 출력하게 돼 네트워크의 표현력을 제한하게 된다는 얘기입니다.

따라서 학습파라메터의 초기값을 잘 설정해주어야 합니다. 이와 관련해 다양한 파라메터 초기화 방법론이 제시되었습니다. 일부를 소개하면 다음과 같습니다. 여기에서 $n_{in}$은 직전 레이어의 차원수, $n_{out}$은 다음 레이어의 차원수입니다. 아래 초기화식은 각 층의 가중치 $W$에 관한 식이고요, 은닉층과 출력층의 편향 $b$는 대개 0으로 설정한다고 합니다.



### LeCun Initialization (Xavier Initialization)

$$W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 1}{ { n }_{ in } } $$



### Glorot Initialization

$$W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 2 }{ { n }_{ in }+{ n }_{ out } }$$



### He Initialization

$$W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 2 }{ { n }_{ in } }$$



## 학습 파라메터 최적화 (1)

학습 파라메터 최적화 기법으로 널리 쓰이는 **그래디언트 디센트(Gradient Descent)**는 기본적으로 아래와 같은 구조를 지닙니다. 여기에서 $θ$는 갱신 대상 학습 파라메터, $dL/dθ$는 Loss에 대한 $θ$의 그래디언트, $η$는 **학습률(leaning rate)**을 의미합니다. 즉 아래 식은 Loss에 대한 $θ$의 그래디언트 반대 방향으로 $η$만큼 조금씩 $θ$를 업데이트하라는 뜻입니다.

$$\theta \leftarrow \theta -\eta \frac { \partial L }{ \partial \theta  }  $$

그래디언트 디센트 방법론에도 여러 변형이 존재합니다. 세 가지 살펴보겠습니다. 아래 제시된 코드는 파이썬 기준입니다.



### Batch Gradient Descent

이 방법은 전체 학습데이터 Loss에 대한 각 파라메터의 그래디언트를 한꺼번에 구한 뒤 1 epoch동안 모든 파라메터 업데이트를 단 한번 수행하는 방식입니다. 매우 느리고 메모리 요구량이 많다는 단점이 있지만 최적해를 찾을 수 있다는 장점이 있다고 합니다. (It's guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces)

```python
for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
```



### Stochastic Gradient Descent

학습데이터의 순서를 랜덤으로 섞은 뒤 개별 레코드(위 코드에서 example) 단위로 Loss와 그래디언트를 구한 뒤 학습 파라메터를 조금씩 업데이트하는 방식입니다. 1 epoch동안 학습데이터 개수만큼의 업데이트가 수행됩니다. BGD보다 훨씬 빠르면서도 수렴 결과가 BGD와 일치(학습률을 줄였을 때)한다고 합니다. 

```Python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
    	params_grad = evaluate_gradient(loss_function, example, params)
    	params = params - learning_rate * params_grad
```



### Mini-batch Gradient Descent

이 방식은 개별 레코드가 아니라 batch_size(아래 코드에서 50) 단위로 학습한다는 점을 제외하고는 SGD와 같습니다. SGD에 비해 안정적으로 학습하는 경향이 있다고 합니다. 게다가 데이터가 배치 단위로 들어가게 되면 사실상 행렬 연산이 되기 때문에 시중에 공개돼 있는 강력한 라이브러리를 사용할 수 있다는 장점 또한 있습니다.

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
    	params_grad = evaluate_gradient(loss_function, batch, params)
    	params = params - learning_rate * params_grad
```





## 학습 파라메터 최적화 (2)

그래디언트 디센트 계열 외에 다양한 최적화 기법을 소개합니다. 최근 각광받고 있는 기법들입니다.



### Momentum

모멘텀은 운동량을 뜻하는 단어로 물리 현상과 관계가 있습니다. 예컨대 아래 그림처럼 공이 한번 움직이기 시작하면 기울기 방향으로 힘을 받아 가속하게 되죠. 모멘텀 기법은 바로 이 점에 착안했습니다.

![](https://i.imgur.com/SQAkzU2.png)


모멘텀 기법은 수식으로는 다음과 같이 쓸 수 있습니다. 여기에서 $μv$는 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 합니다($μ$는 0.9 등의 값으로 설정합니다). 물리에서는 지면 마찰이나 공기 저항에 해당합니다. 나머지는 그래디언트 디센트 기법과 동일합니다.

$$v\leftarrow \mu v-\eta \frac { \partial L }{ \partial \theta  } \\ \theta \leftarrow \theta +v$$

모멘텀 기법의 최적화 효과를 직관적으로 나타낸 그림은 아래와 같습니다. 하단 좌측 그림을 보시면 현재의 그래디언트가 모멘텀과 같은 방향이라면 업데이트가 더 크게 이뤄지게 됩니다. 하단 우측 그림에서 최적화 지점이 원 내부 중앙이라고 했을 때 모멘텀 기법이 조금 더 효율적인 업데이트 경로를 거치고 있는 점을 확인할 수 있습니다.

![](https://i.imgur.com/6nnAWF8.png)

코드로는 아래와 같습니다. 여기에서 $μ$는 사용자가 지정하는 하이퍼파라메터입니다.

```python
param_grad = evaluate_gradient(loss_function, data, params)
v = mu * v - learning_rate * param_grad
param = v + param
```



### Nesterov Accelerated Gradient

이 기법은 모멘텀 기법을 업그레이드한 버전입니다. 현재 학습 파라메터(붉은색 원 : 아래 코드에서 $params$)를 직전까지 축적된 그래디언트 방향(녹색선)으로 이동시킵니다. 이 벡터($params_{ahead}$)를 기준으로 그래디언트(붉은색 선 : $params\_grad_{ahead}$)를 계산합니다. 실제 업데이트는 둘을 모두 반영해 이뤄집니다. 

> **모멘텀과의 차이** : 모멘텀은 현재 점(붉은색 원)에서 그래디언트를 구합니다. NAG는 녹색선과 빨간선이 이루는 꼭지점에서 그래디언트를 구합니다.

![](https://i.imgur.com/xC1YRNZ.png)


이 기법을 코드로 나타내면 아래와 같습니다. 여기에서도 $μ$는 역시 사용자가 지정하는 하이퍼파라메터입니다.

```Python
params_ahead = params + mu * v
params_grad_ahead = evaluate_gradient(loss_function, data, params_ahead)
v = mu * v - learning_rate * params_grad_ahead
params = v + params
```



### AdaGrad

**학습률 감소**와 연관된 기법입니다. AdaGrad는 각각의 학습 파라메터에 맞춤형으로 학습률을 조정하면서 학습을 진행합니다. 수식은 아래와 같습니다. 여기에서 $⊙$는 행렬의 원소별 곱셈을 의미합니다. 식을 보시면 학습 파라메터의 원소 가운데 많이 움직인(크게 갱신된) 원소는 학습률이 낮아지게 돼 있습니다. 다시 말해 학습률이 학습 파라메터의 원소마다 다르게 적용된다는 뜻입니다.

$$h\leftarrow h+\frac { \partial L }{ \partial \theta  } \odot \frac { \partial L }{ \partial \theta  } \\ \theta \leftarrow \theta -\frac { \eta  }{ \sqrt { h }  } \frac { \partial L }{ \partial \theta  } $$

이를 코드로 나타면 아래와 같습니다. 아래 코드에서 eps는 분모가 너무 0에 가깝지 않도록 안정화하는 역할을 합니다. 보통 $10^{-4}$에서 $10^{-8}$의 값을 쓴다고 합니다.

```Python
params_grad = evaluate_gradient(loss_function, data, params)
h = h + params_grad**2
params = params - learning_rate / (np.sqrt(h) + eps) * params_grad
```



### RMSProp

AdaGrad는 학습률 $η$를, 과거의 기울기를 제곱한 값을 계속 더해나간 $h$로 나눠줍니다. 학습을 진행할 수록 $η$가 지속적으로 작아진다는 뜻입니다. 계속 학습하면 $η$가 0이 돼서 학습이 불가능해지는 시점이 옵니다. RMSProp은 이를 개선하기 위한 기법입니다. 즉 과거의 모든 기울기를 다 더해 균일하게 반영하는 것이 아니라, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영하기 위해 $h$를 계산할 때 **지수이동평균(Exponential Moving Average)**을 적용합니다. 

코드는 아래와 같습니다. 여기에서 decay_rate는 사용자 지정 하이퍼파라메터이고 보통 $[0.9, 0.99, 0.999]$ 가운데 하나를 쓴다고 합니다.

```python
params_grad = evaluate_gradient(loss_function, data, params)
h = decay_rate * h + (1 - decay_rate) * params_grad**2
params = params - learning_rate / (np.sqrt(h) + eps) * params_grad
```



### Adam

모멘텀은 공이 구르듯 하는 물리 법칙에 착안해 만들어진 기법입니다. AdaGrad과 RMSProp은 학습 파라메터의 개별 원소마다 학습률을 달리 적용합니다. 두 기법을 합친 것이 바로 Adam입니다. 퍼포먼스가 좋아서 최근 많은 관심을 받고 있는 기법인데요. 코드는 다음과 같습니다.

```python
params_grad = evaluate_gradient(loss_function, data, params)
m = beta1 * m + (1 - beta1) * params_grad
v = beta2 * v + (1 - beta2) * params_grad**2
params = params - learining_rate * m / (np.sqrt(v) + eps)
```

여기에서 $beta_1$, $beta_2$, $eps$는 사용자가 지정하는 하이퍼파라메터입니다. 논문에 따르면 각각 0.9, 0.999, $10^{-8}$이 좋다고 합니다.



### 각 기법 비교

너무나도 유명한 그림이라 설명은 생략하겠습니다. 저 역시 정리 용도로 올려 둡니다.

![](https://i.imgur.com/U34fEr3.gif)

![](https://i.imgur.com/i98ywya.gif)

















