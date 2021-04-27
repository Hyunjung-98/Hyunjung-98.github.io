---
title: "[논문리뷰] Generative Adversarial Nets"
categories:
    - 논문리뷰
tags:
    - GAN
    - paper review
    - Computer Vision
use_math: true
---

최초로 GAN 모델을 제안한 논문을 리뷰하면서 GAN에 대해 공부해봤다. 생각보다 금방 이해가 되지 않아, 여러 포스팅을 참고하면서 차근차근 읽었다.

## Abstract

두 개의 adverserial한 모델을 학습하여 generative models을 estimate하는 방법 제시하였다.

두 가지 모델은 다음과 같다.

1. a generative model G
    - 데이터의 분포를 알아내 이미지를 만드는 함수
    - D가 잘못 예측할 확률을 높이는 방향, 즉 D(G(z)) = 1이 되는 방향으로 학습
2. a discriminative model D
    - sample이 G가 아닌 training data에서 생성됐을 확률을 추정
    - training data에서 뽑은 sample x일 경우 $D(x) =1$, 임의로 생성된 노이즈 z로 생성된 sample일 경우 → $D(G(z)) = 0$ 이 되는 방향으로 학습

## Introduction

기존 DL은 backpropagation & dropout을 바탕으로 한 discriminative model 위주로 발전하였다고 한다.

당시에 Generative model의 발전은 미약했는데, 이 논문에서는 backprop과 dropout만으로 G모델과 D모델을 학습하는 방법을 제시한다.

(approximate inference or Markov Chains 필요X)

## Adversarial nets

- (두 모델 모두 multilayer perceptron으로 설정)

GAN 알고리즘의 loss function이자 objective function인 V(D,G)는 다음과 같다.

$min_Gmax_DV(D,G)=E_{x∼p_{data}(x)}[logD(x)]+E_{z∼p_z(z)}[log(1−D(G(z)))]$

V(D, G)를 최소화하는 G, V(D,G)를 최대화하는 D를 찾는 것이 모델 학습의 목적이다.

여기서, notation에 대해 살펴보면 다음과 같다.

- $p_{data}$: training data의 분포
- $p_g$: $G$로 인해 생성된 데이터의 분포
- $z$: 임의로 생성된 noise
- $p_z$: noise variables의 분포
- $G(z)$: noise $z$로 생성한 데이터.  z를 매핑하여 x로 만드는 함수.
- $x$: 실제 training 데이터($~p_{data}(x)$이므로!)
- $D(x)$: 주어진 sample $x$가 $p_g$가 아닌 $p_data$에 있을 확률. output은 확률값인 scalar.

여기서 두 모델 D, G가 각각 원하는 것에 대해 살펴보자.

1. D가 원하는 것
    - $x \~ p_{data}$일 때, D(x)=1이 되길 원함 (⇒ log(1) = 0)
    - 이 때, 노이즈로 생성한 데이터 G(z)에 대해 D(G(z)) = 0이 되길 원함(⇒ log(1-0)=0)
    - V = 0일 때, D의 입장에서 V의 최댓값을 얻을 수 있음
2. G가 원하는 것
    - D(x)의 성능에는 관심이 없고, D(G(z))=1인 것에만 관심을 가짐
    - 노이즈로 생성한 데이터 G(z)에 대해 D(G(z)) = 1이 되길 원함 (⇒ log(1-1)=-∞)
    - V = -∞일 때, G의 입장에서 V의 최솟값을 얻을 수 있음

논문에서의 그림으로 G와 D의 학습 방향에 대해 이해보면 다음과 같다.

![/assets/images/posts/2021-04-23/Untitled.png](/assets/images/posts/2021-04-23/Untitled.png)

여기서 아래 두 선은 z가 G를 통해 x로 매핑되는 것을 보여주고,

검은 점선이 실제 데이터의 분포인 $p_{data}$, 초록 실선이 새로 만들어진 데이터의 분포인 $p_g$, 파란 점선이 데이터가 $p_{data}$에서 온 것인지에 대한 분포를 의미한다.

- (a): 학습 초기. $p_d$와 $p_{data}$가 전혀 다르게 생김.
- (b), (c): 학습 진행 중. 파란점선인 D가 실제 training data에서 온 데이터를 더 잘 예측하도록 학습되었기 때문에 더 스무스해짐. $p_d$와 $p_{data}$의 분포가 점점 닮아감.
- (d): 학습 결과. $p_d$와 $p_{data}$가 동일해짐. 실제 training data와 G(z)가 너무 비슷하여 sample이 어디서 온 데이터인지 구분할 수 없게 되므로 D(x)=0.5가 됨.

따라서 학습의 최종 목표는 **p_d = p_{data}**가 되도록 하는 것, 즉 **D(x) = 0.5**가 되게 하는 것임을 알 수 있다.

학습은 다음의 순서를 반복한다.

1. k steps만큼 D를 optimize
2. 1 step만큼 G를 optimize

학습 초기에는 G의 성능이 좋지 않아 D가 높은 신뢰도로 reject할 수 있다. 이 때, $log(1-D(G(z)))$ 로 계산하면 학습이 매우 느려진다. 따라서 $log(1-D(G(z)))$를 minimize하는 방향보다는, $log(D(G(z)))$를 maximize하는 방향으로 G 학습하여 학습 속도를 높인다.

## Theoretical Results

### 1. global optimum은 $p_g = p_{data}$인 위치에서 생김

증명을 하기위해 하나의 명제에 대해 알아본다.

![/assets/images/posts/2021-04-23/Untitled%201.png](/assets/images/posts/2021-04-23/Untitled%201.png)

이 명제에 대한 증명은 다음과 같다.

![/assets/images/posts/2021-04-23/Untitled%202.png](/assets/images/posts/2021-04-23/Untitled%202.png)

- $a = p_{data}(x)$, $b = p_{g(x)}$로 설정을 한다면, 위의 전제가 성립함을 확인할 수 있다.

이 명제를 $C(G) = max_DV(G,D)$에 적용하면, 다음이 성립함을 확인할 수 있다. 

![/assets/images/posts/2021-04-23/Untitled%203.png](/assets/images/posts/2021-04-23/Untitled%203.png)

이를 바탕으로 Thm1인 

"The global minimum of the virtual training criterion C(G) is achieved if and only if pg = pdata. At that point, C(G) achieves the value 􀀀log 4."

을 보일 수 있다.

$p_g = p_{data}$일 때, $D*_G(x)=1/2$임을 위에서 확인하였다. 

$D*_G(x)=1/2$라고 한다면, 위의 식(4)에 의해 $C(G) = -log4$임을 확인할 수 있다.

이때의 값이 C(G)의 global minimum임을 확인하기 위해 $D*_G(x)=1/2$일 때의 식(4)를 변형하여 풀어주면 다음과 같다. 

![/assets/images/posts/2021-04-23/Untitled%204.png](/assets/images/posts/2021-04-23//Untitled%204.png)

KL(Kullback-Leibler divergence)와 JSD(Jensen-Shannen Divergence)의 개념이 생소하여, [Hyeongmin Lee님의 포스트](https://hyeongminlee.github.io/post/prob002_kld_jsd/)의 도움을 받아 이해했다.

$KL(p_{data}|\frac {p_{data}+p_g}{2} ) = E[log(p_{data})-log(\frac {p_{data}+p_g}{2})] = E[log(2*pdata/(pdata+pg))] = log2 + E[log(pdata/(pdata+pg))]$ 이므로 위의 식이 성립함을 알 수 있다.

JSD는 항상 양수이고 두 distribution이 일치할 때만 0이므로 C∗=−log(4)가 C(G)의 global minimum이며 그 유일한 해가 pg=pdata임을 알 수 있다.

### 2. Algorithm1의 수렴

알고리즘1은 다음과 같다. 

![/assets/images/posts/2021-04-23/Untitled%205.png](/assets/images/posts/2021-04-23//Untitled%205.png)

D에 관해 mini-batch 크기 m만큼 gradient ascending을  k번 실시한 후, G에 관해  mini-batch 크기 m(새로운 m)만큼 gradient descending을 해주는 것을 반복한다.

![/assets/images/posts/2021-04-23/Untitled%206.png](/assets/images/posts/2021-04-23//Untitled%206.png)

Thm2을 증명하기 위한 개념들을 간단히 살표보면 다음과 같다.

- Supremum: [0, 2)의 경우, 2, 3, 124.2 모두 상계(upper bound)가 될 수 있다. 이 중 2를 상한(supremum, least upper bound)이라고 부른다.
- Subgradient

    ![/assets/images/posts/2021-04-23/Untitled%207.png](/assets/images/posts/2021-04-23/Untitled%207.png)

이 개념을 기억하며, thm2를 증명해자.

![/assets/images/posts/2021-04-23/Untitled%208.png](/assets/images/posts/2021-04-23/Untitled%208.png)

(약간의 논리적 오류가 있는듯)

## Experiments

- 학습 데이터: MNIST, TFD, CIFAR-10
- Activation function
    - G nets: rectifier linear activations, sigmoid activations
    - D net: maxout activations
- D net 학습 시 dropout 사용
- G net 학습 시 가장 아래의 layer에만 noise 사용

## References

- 논문: [Generative Adversarial Nets - Ian Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661)
- 수식 이해에 도움을 받은 포스팅: [http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)

    [https://www.youtube.com/watch?v=kLDuxRtxGD8](https://www.youtube.com/watch?v=kLDuxRtxGD8)

- 전체 흐름에 대한 도움을 받은 포스팅: [https://blog.naver.com/euleekwon/221558014002](https://blog.naver.com/euleekwon/221558014002)
- KL, JSD 공부에 도움을 받은 포스팅: [https://hyeongminlee.github.io/post/prob002_kld_jsd/](https://hyeongminlee.github.io/post/prob002_kld_jsd/)
- Supremum 공부에 도움을 받은 포스팅: [https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221107601176&proxyReferer=https:%2F%2Fwww.google.com%2F](https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221107601176&proxyReferer=https:%2F%2Fwww.google.com%2F)
- [https://wikidocs.net/18963](https://wikidocs.net/18963)