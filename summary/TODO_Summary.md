## 1. 구현 Summary

### 1.1 ImageNet Implementation

#### Architecture

![table1](../pics/table1.png)

**Table 1.** 이미지넷을 위한 아키텍처. 구축된 블락들은 쌓이는 블락의 갯수와 함께 대괄호로 묶여 있습니다. Down sampling은 conv3_1, conv4_1, conv5_1 에서 stride 값을 2로 주며 진행되었습니다.

#### Augmentation

- Scale augmentation을 위해 이미지는 짧은 쪽이 [256, 480]사이의 랜덤한 값을 갖도록 resize
- 이미지와 이를 수평으로 뒤집은 이미지에 대해 224X224 random crop이 진행되고 각 픽셀별 평균을 빼주기
- 기본적인 color augmentation

#### Implementation detail

| Implementation | Detail |
| ------------------ | ------------------------------------------------- |
| Batchnorm          | 각 convolution 이후, 활성화 함수 적용 이전에 진행 |
| Weight Initializer | He initializer                                    |
| Batchsize          | 256                                               |
| Optimizer          | SGD, momentum = 0.9                               |
| Learning rate      | 0.1부터 시작하고 에러가 줄지 않을때 10씩 나눔     |
| Weight decay | 0.0001 |

### 1.2 CIFAR-10 Implementation

#### Architecture

모두 projection shortcuts가 아닌 identity shortcuts를 사용했습니다.  Downsampling은 stride 값을 2로 줌으로써 진행했습니다.

n 을 {3, 5, 7, 9, 18, 200}로 설정해 20, 32, 44, 56, 110, 1202 레이어 네트워크로 비교

![cifiar10-architecture](../pics/cifiar10-architecture.png)

#### Augmentation

- 각 사이드별로 4 픽셀씩 패딩을 진행하고 패딩된 이미지와 수평으로 flip된 이미지에서 32X32 random sample crop을 진행했습니다.
- Testing 단계에서는 원본 32X32 이미지만을 사용했습니다.

#### Implementation detail

| Implementation     | Detail                                             |
| ------------------ | -------------------------------------------------- |
| Batchnorm          | 각 convolution 이후, 활성화 함수 적용 이전에 진행  |
| Weight Initializer | He initializer                                     |
| Batchsize          | 256                                                |
| Optimizer          | SGD, momentum = 0.9                                |
| Learning rate      | 0.1부터 시작하고 32K, 48k iteration 마다 10씩 나눔 |
| Weight decay       | 0.0001                                             |
| Train/Val split      | 45k / 5k |

## 2. 내용 Summary

당시에 많은 연구들로부터 컨볼루션 네트워크의 깊이가 깊어지면 깊어질수록 추출할 수 있는 더 풍부한 feature를 추출할 수 있다는 점에서 네트워크의 "깊이"가 매우 중요해졌습니다.

깊이가 중요해짐에 따라, 하나의 의문이 생깁니다. 단순히 많은 레이어들을 쌓기만하면 더 좋은 네트워크를 구성할 수 있을까요?

답은 아니오입니다. Vanishing/exploding gradient 현상으로 인해 네트워크의 시작부터 수렴이 잘 되지 않았습니다. 여기서 수렴은 올바른 방향으로의 학습을 의미합니다.

Vanishing/exploding 현상은 batch normalization이나 He initialization등의 노력을 통해 해결할 수 있게되었습니다. 위 문제를 해결하면서 네트워크가 수렴을 시작할 수 있게되었습니다.

더 깊은 네트워크가 수렴을 시작할 수 있게 되었을때, "Degradation" 문제가 발생했습니다.

일반적으로 층을 겹겹이 쌓아 구성하는 네트워크의 층을 더 깊게 쌓았더니 오히려 Training error가 더 높은, 즉 학습 부분에서의 저하(degradation)가 일어나는 현상을 Degradation problem이라고 합니다.

![figure1](../pics/figure1.png)

위 이미지의 왼쪽그림을 보면 56개의 레이어를 갖는 네트워크의 에러가 20개의 레이어를 갖는 네트워크의 에러보다 더 높은 것을 확인할 수 있습니다. 

이 문제를 해결하기 위해 ResNet 논문에서는 degradation 문제가 없는 깊은 네트워크를 구성하기 위해 "Redisual Network"라는 구조적인 솔루션을 제안합니다. 

Residual Network에서는 일반적으로 평평하게 쌓인 레이어들에 Residual mapping을 추가해 학습을 진행합니다. 추가되는 매핑 파트는 Identity mapping으로 인풋값을 그대로 매핑합니다.

- Original mapping : $H(x)$ → $F(x) + x$
- Residual mapping : $F(x) := H(x) - x$

만약 다수의 비선형 레이어들이 임의의 복잡한 함수에 점근적으로 근사할 수 있다고 가정했을때, 다수의 비선형 레이어들은 residual 함수에도 점근적으로 근사할 수 있을 것입니다

쌓여져있는 레이어들이 H(x)에 근사하도록 하는 것 대신에, ResNet에서는 레이어들이 residual 함수인 $F(x) := H(x) − x$ 에 근사하도록 했습니다.

ResNet에서는 일반적인 평평한 형태로 쌓인 레이어($$H(x)$$)보다 매핑의 결과와 Input의 차이를 계산하는 Residual 매핑($$F(x)$$)을 최적화하는게 더 쉽다고 가정했습니다.

![fig2](../pics/figure2.png)

ResNet에서 수식 $F(x) + x$를 "shortcut connection"으로 구현했습니다.

Shortcut connections는 하나 또는 하나 이상의 레이어를 건너 뛰는 연결을 말합니다. Resnet의 경우, shortcut connections는 단순히 identity mapping을 수행하고, 그 아웃풋은 쌓여져 있는 레이어의 아웃풋에 더해집니다.

ResNet 논문에서는 위 Residual Network를 ImageNet 데이터셋과 CIFAR-10 데이터셋에서 다양한 변화를 주며 실험을 진행했습니다. 각 데이터셋에서 Residual mapping을 진행한 ResNet 네트워크와 VGG 형태처럼 단순히 쌓는 구조인 Plain 네트워크를 비교합니다.

ResNet 네트워크가 Plain 네트워크보다 훨씬 높은 성능을 보이고, Residual mapping을 진행한 경우 degradation 문제도 관측되지 않았습니다. 즉 ResNet은 깊이가 깊어질수록 더 높은 정확도의 모델을 얻는 것이 가능합니다.





