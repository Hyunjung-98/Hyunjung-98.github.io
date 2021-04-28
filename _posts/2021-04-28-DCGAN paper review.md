---
title: "[논문리뷰] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
categories:
    - 논문리뷰
tags:
    - GAN
    - paper review
    - Computer Vision
use_math: true
---

현재 연구되는 응용 GAN 모델들 중 가장 기본이 되는 모델인 DCGAN을 제시한 논문을 리뷰하였다. DCGAN은 기존 FC-layers로 구성된 GAN의 구조를 특정 architecture를 가지는 CNN으로 변경한 모델이다.


## Abstract

지도학습 분야에서 CNN의 연구는 활발했으나, 비지도학습에서 CNN의 연구는 활발하지 않았었다. 이 논문은 DCGANs를 소개함으로써 비지도학습에서의 CNN 연구를 진행하였다. 연구된 모델은 generator과 discriminator 모두에서 object과 scene을 잘 잡아내는 성능을 보였으며, 학습된 모델을 이미지 분야의 다른 태스크에서도 적용할 수 있다고 한다.

## 1. Introduction

대용량의 unlabeld data로부터 reusable feature representaitons를 배워 supervised learning에 적용하는 연구는 활발하게 진행되어왔다. CV 분야에서는 GAN을 통해 이를 구현할 수 있었다. 하지만 GAN은 학습하기 불안정하고, generator가 무의미한 output를 내는 경우가 있는 등에 한계가 있어 GAN을 시각화하는 연구는 활발하지 않았다. 

이 논문에서는,

- 학습하기 안정적인 Convolutional GANs의 architecture (DCGANs)을 제시하고,
- 학습된 discriminator을 image classification 분야에 적용하고,
- GAN을 통해 학습된 filter를 시각화하고, 특정 object를 학습한 filter를 보여주고,
- generators가 생성된 샘플의 semantic qualities의 작동을 쉽게 하는 vector arithmetic 특징을 가짐을 보인다.

## 2. Related Work

### 2.1 Representation Learning from Unlabeled Data

이 분야에서의 진행된 연구는 다음과 같다.

- Classical approach는 K-means를 통한 data clustering을 하여 classification score를 향상시키는데 이 clusters를 사용하는 것
- 이미지에서는 hierarchical clustering
- Auto-encoders 등을 통해 학습하는 것
- Deep belief networks

### 2.2 Generating Natural Images

이 분야는 크게 parametirc, non-parametric의 두 카테고리로 나뉜다.

Parametric 분야에서 GAN의 등장이 있었으나, noisy하고 incomprehensible한 이미지가 생성된다는 한계가 있었다.

## 3. Approach and Model Architecture

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled.png](/assets/images/posts/2021-04-28/Untitled.png)

1. Pooling layers를 **discriminator에는 strided convs, generator에는 fractional-strided convs**로 대체

    deterministic spatial pooling을 대신하는 all convolutional net은 strided convolutions와 같이 기능하며, network가 own spatial downsampling(이미지의 크기(픽셀)을 줄이는 것)을 학습하도록 한다. 

    이를 generator에 활용하여 own spatial upsampling가 discriminator를 학습하도록 한다.

    > Strided Convolutions / Fractional-Strided Convolutions(Transposed Convs)
    ![strided](/assets/images/posts/2021-04-28/strided.gif)
    ![fractional_strided](/assets/images/posts/2021-04-28/fractional_strided.gif)

    - 파란색이 input, 초록색이 output
    - fractional-strided convs에서는 output의 크기가 input보다 커짐
2. Conv features 위의 FC-layers를 제거
3. Batch Normalization을 이용하여 학습을 안정화
4. Ouput layer를 제외한 generator의 모든 layers에 ReLU 사용. Output lyaer에는 Tanh 사용.
5. Discriminator의 모든 layers에 LeakyReLU 사용

## 4. Details of Adversarial Training

**<학습 조건>**

- Dataset
    1. LSUN: 침실 사진. 3M images
    2. Faces: 사람 얼굴 사진. 3M images form 10K people
    3. ImageNet-1k
- mini-batch size = 128인 SGD 환경에서 학습
- weights initialization: $N(0, 0.02)$
- LeakyReLU의 slope = 0.2
- Adam optimizer 사용
- learning rate = 0.0002
- momentum term $\beta_1$=0.5

## *Result

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled%201.png](/assets/images/posts/2021-04-28/Untitled%201.png)

Epoch=1일 때 생성된 bedroom이미지. (LSUN dataset 활용)

learning rate를 작게 설정하고, mini-batch SGD를 사용하였기 때문에 training set의 이미지를 복제한 것이 아닌 새로 학습된 generator로 생성한 것임을 알 수 있다. 

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled%202.png](/assets/images/posts/2021-04-28/Untitled%202.png)

Epoch=5일 때 생성된 bedroom 이미지. (LSUN dataset 활용)

반복된 noise textures로 인해 여러 샘플에서 baseboard(걸레받이)가 반복되게 나타난 것을 보아 underfitting이 발생한 것을 확인할 수 있다.

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled%203.png](/assets/images/posts/2021-04-28/Untitled%203.png)

각 행마다 Z값(latent space)을 조금씩 변경했을 때 smooth transition이 발생한 것을 확인할 수 있다. 

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled%204.png](/assets/images/posts/2021-04-28/Untitled%204.png)

Discriminator 이 학습한 feature을 시각화한 이미지. 각 input에서 어떤 feature를 활성화하는지 시각화하여 보여준다.

![Unsupervised%20Representation%20Learning%20with%20Deep%20Con%20a9d8c2e67dfe40ee9847da91a02cb7e4/Untitled%205.png](/assets/images/posts/2021-04-28/Untitled%205.png)

Generator에서 사용되는 latent spzce Z에서의 산술연산을 통한 object manipulation

## References

- [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
- [DCGAN 논문 이해하기](https://angrypark.github.io/generative%20models/paper%20review/DCGAN-paper-reading/)
- [초짜 대학원생의 입장에서 이해하는 Deep Convolutional Generative Adversarial Network (DCGAN) (2)](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-2.html)