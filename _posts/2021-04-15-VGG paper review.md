---
title: "[논문리뷰] Very Deep Convolutional Networks for Large-Scale Image Recognition"
categories:
    - 논문리뷰
tags:
    - CNN
    - paper review
    - Computer Vision
use_math: true
---

AlexNet 논문 리뷰에 이어 AlexNet 이후의 CNN 모델인 VGG 논문을 리뷰하였다.

## Abstract

- large-scale image recognition에서 convnet depth의 영향 연구
- __small (3x3) conv filters를 이용하여 depth를 16-19 layers까지 늘림__

## 1. Introduction

- ImageNet등의 large public image repo과 GPU 등의 computing systems 발달 ⇒ CNN의 성능 향상
- original CNN architecture에서 정확도를 높이기 위한 시도가 있었음
    - 1st conv-layer에서 window size와 stride 크기 줄이기
    - whole image와 multiple scale를 사용하여 train & test
- 이 논문에서 매우 작은 conv filters(3x3)를 모든 레이어에 적용

    ⇒ ILSVRC classification & localisation tasks에서 sota 달성

    ⇒ 다른 데이터셋어서도 좋은 성능 보임

## 2. ConvNet Configuration

### 2.1 Architecture

- ConvNet의 input image size는 224x224x3(RGB)로 고정
- training image 각 pixel에 대해 mean RGB 값을 빼줌
- left/right, up/down, center를 잡기위한 최소 filter size인 3x3 사용
- 일부 layer에서 1x1 conv filter 사용 (input channels의 linear transformation)
- conv stride = 1
- padding = 1 pixel for 3x3 layers
- five max-pooling layers. window size = 2x2. s =2
- 3 FC layers
    - 1st, 2nd layers: 4096 channels
    - 3rd(softmax) layer: 1000 channels
- activation function: ReLU
- Local Response Normalisation X (성능 향상 없음)

### 2.2 Configurations

![Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20I%2050eb8f02368e4b7f944b226b1e23d92f/Untitled.png](/assets/images/posts/2021-04-15/0.png)

- 'A-E'의 각 network는 depth만 다르고, 위의 2.1에서 언급한 configurations를 따름
- A → E로 갈수록 depth 증가
- conv-layers의 width(# of channels)는 1st layer에서 64로 시작하여, 각 max-pooling layer를 거칠 때마다 512가 되기 전까지 2배씩 증가함

### 2.3 Discussion

- 기존의 논문들에서 첫 번째 레이어에서 비교전 큰 filters를 사용한 것에 비해, 이 논문에서 3x3의 filters를 전체 레이어에 사용
- 3x3 conv-layer 2개를 쌓는 것이 5x5 conv-layer 1개를 사용하는 것보다 효과적임
- 3x3 conv-layer 3개가 7x7 conv-layer 1개의 효과를 지님
    - making decision function more discriminative
    - decreasing the number of parameters
- 1x1 conv layers
    - increasing the non-linearity of the decision function w/o affecting the receptive filed of the con-layers

## 3. Classification Framework

### 3.1 Training

- 기존 AlexNet 논문에서처럼 training은 momentum이 있는 mini-batch gradient descent를 이용한 multinomial logistic regression objective를 최적화하면서 진행
- batch size = 256. momentum = 0.9
- Regularizaion
    - weight decay (L2 penalty)
    - 1st, 2nd FC-layers에서 0.5의 ratio로 dropout
- learning rate: 초기값 $10^{-2}$로 설정. validation set accuracy에 변동이 없으면 10배씩 감소. 총 3번 감소됨.
- 총 370K iters (74 epochs)
- 기존 Alexnet 논문보다 paremeters 수가 더 많고 더 깊음에도 불구하고 convergion까지의 epoch 수 더 적음
    - grater depth와 smaller conv filter size에 대한 implicit regularisation
    - 일부 layer에서의 pre-initialisation
- Initialisation
    - random initialization으로 configuration A부터 train
    - 더 깊은 네트워크를 train할 때 첫 4개의 conv-layers와 마지막 3개의 FC-layers를 configuartion A의 값으로 초기화. (중간의 layers는 randomly initalized)
    - pre-initialized layers에 대해서는 learning rate를 감소시키지 않음
    - random initialization은 weights$~N(0, 10^{-2})$를 따르도록 함.
    - biases는 0으로 초기화
- input data
    - 224x224의 input images를 얻기 위해 각 SGD iteration마다 training image를 randomly crop함
    - 추가적인 train set augmentation을 위해 random horizontal flipping과 random RGB colour shift 사용
- **Training Image size**
    - S: isotropically-rescaled training image의 가장 작은 부분. training scale.
    - crop size가 224x224이므로, 224 이상의 모든 값은 S가 될 수 있음
    - S=224: crop은 이미지 전체를 잡을 것임
    - S>224: crop은 object part를 포함하는 이미지의 일부를 잡을 것임
    - S를 설정하는 두 가지 방법
        1. single-scale traing. S를 모든 이미지에 대해 고정함. 논문에서는 우선 S=256으로 설정한 뒤, S=384로 설정함. S=384로 설정했을 때의 training 속도를 높이기 위해 S=256으로 설정했을 때의 weights로 intialize하고, learning rate를 $10^-3$으로 줄임.
        2. multi-scale training. 각 이미지의 S를 [S_min, S_max] 범위의 랜덤한 값으로 설정함. 이 논문에서는 [256, 512] 범위의 값을 사용함. 이미지마다 object 크기가 다를 수 있으므로, training 시 이를 고려함. (training set augmentation by scale jittering) 속도를 높이기 위해 S=384로 pretrain된 모델에 multi-scale을 적용함.

### 3.2 Testing

- 이미지를 pre-defined smalleset image side (test scale) Q로 isotropically rescale
- test scale Q는 train scale S와 달라도 괜찮음
- FC-layers가 conv-layers로 변환됨 (1st FC-layer → 7x7 conv-layer. 2nd, 3rd FC-layers → 1x1 conv-layer)
- 네트워크를 전체 이미지에 적용하여 class수만큼의 channel을 갖는 class score map 생성
- a fixed-size vector를 얻기 위해 class score map은 spatially averaged
- 이미지를 horizontal fipping하여 augmetation 진행
- 이후 원본 이미지와 flipped image의 softmax 값을 평균내여 final score 계산

## 4. Classification Experiments

- Dataset
    - ILSVRC-2012
    - images of 1000 classes
    - training 1.3M / validation 50K (→ test set으로 사용) / testing 100K(label X)
- 평가지표: top-1 error & top-5 error

### 4.1 Single-scale Evaluation

- test image size
    - for fixed $S$ ; $Q = S$
    - for jittered $S \in [S_min, S_max]$ ; $Q = (S_min + S_max)$

![Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20I%2050eb8f02368e4b7f944b226b1e23d92f/Untitled%201.png](/assets/images/posts/2021-04-15/1.png)

- Local response normalisation 사용 X
- depth가 증가하면 error 감소 : A → E로 갈수록 error 감소
- additional non-linearlity 효과O : C 성능 > B 성능. C는 B에서 3개의 1x1 conv-layers 추가한 모델
- non-trivial filters를 이용하여 spatial context 포착: D 성능 > C 성능. D는 C의 3개의 1x1 conv-layers를 모두 3x3 conv-layers로 변경한 모델
- deep network w/ small filters > shallow network w/ large filters: 10개의 3x3 conv-layers를 가진 B와 5개의 5x5 conv-layers를 가진 모델을 비교했을 때, B의 성능 더 좋음
- multi-scale training이 single-scale training보다 성능이 좋음

### 4.2 Multi-scale Evaluation

- test image
    - 여러 rescaled version으로 train 후 class 평균 결과 산출
    - for fixed $S$ ; $Q = {S-32, S, S+32}$
    - for jittered $S \in [S_min, S_max]$ ; $Q = {S_min, (S_min + S_max)/2, S_max}$

![Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20I%2050eb8f02368e4b7f944b226b1e23d92f/Untitled%202.png](/assets/images/posts/2021-04-15/2.png)

- Scale jittering 후 성능 향상

### 4.3 Multi-crop Evaluation

![Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20I%2050eb8f02368e4b7f944b226b1e23d92f/Untitled%203.png](/assets/images/posts/2021-04-15/3.png)

- multi-crop evaludation 성능 > dense evaluation 성능
- multi-crop evaludation과 dense evaluation의 softmax 평균 결과 산출한 결과의 성능이 가장 좋음 (complementary)

### 4.4 ConvNet Fusion

여러 모델의 class 평균 결과를 사용하면 성능 향상

## 5. Conclusion

ConvNet에서 depth를 깊게 하는 것이 정확도 향상에 도움이 됨.

## References
- 논문: [Very Deep Convolutional Networks for Large-Scale Image Recognition][link]]

[link]: https://arxiv.org/pdf/1409.1556.pdf "VGG"