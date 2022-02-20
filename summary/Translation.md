# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift : 2015

## 0. Abstract
Training Deep Neural Networks is complicated by the fact
that the distribution of each layer’s inputs changes during
training, as the parameters of the previous layers change.

> 심층 신경망의 훈련은 이전 layer의 매개 변수가 변경됨에 따라, 훈련 중에 각 layer의 입력 분포가 변경된다는 사실 때문에 복잡하다.

This slows down the training by requiring lower learning
rates and careful parameter initialization, and makes it `notoriously`
hard to train models with `saturating nonlinearities`.

> learning rate를 낮추거나 초기 파라미터를 설정을 조심스럽게 하면, 학습속도가 늦으지머,
> 이 경우는 포화 비선형성을 가진 모델을 학습시키는 것을 어렵게 만든다.

We refer to this phenomenon as `internal covariate
shift`, and address the problem by normalizing layer inputs.

> 우리는 이 상황을 `internal covariate shift` 라고 부르며, 레이어의 입력값을 정규화
> 함으로써 우리는 이 문제를 해결한다.

Our method draws its strength from making normalization
a part of the model architecture and performing the
normalization for each training mini-batch.

> 우리의 방식은 모델의 각 파트와, 각각의 미니배치단위들을 정규화를 진행한다.

Batch Normalization allows us to use much higher learning rates and
be less careful about initialization.

> 배치정규화를 사용하면 매우 높은 learning rate를 사용할 수 있게 해주며, 
> 초기값 생성에 덜 신경을 쓰게 해주는 역할을 한다.
 
It also acts as a regularizer, in some cases eliminating the need for Dropout.

> 이것은 또한 정규화의 기능을 수행함으로써, DropuOut의 필요성을 줄여준다.

Applied to a state-of-the-art image classification model,
Batch Normalization achieves the same accuracy with 14
times fewer training steps, and `beats the original model`
by a `significant margin`. 

> SOTA를 달성한 이미지 분류모델에 적용함으로써, 같은 정확도지만 14번의 학습 step을 줄였으며,
> 그리고 기존 모델의 성능을 크게 앞질렀습니다.

Using an ensemble of batch normalized networks, we improve upon the best published
result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.

> 배치정규화 앙상블을 적용하여, 우리는 ImageNet 분류기의 성능을 향상시켰다.
> : top-5 검증데이터셋의 error가 4.9%로 인간이 평가하는 것을 뛰어넘었다.

---
## 1. Introduction


Deep learning has dramatically advanced the state of the art in vision, speech, and many other areas.

> 딥러닝은 드라마틱하게 비젼, 음성 등 다양한 분야에서 발전중이다.

Stochastic gradient descent (SGD) has proved to be an effective way of training deep networks, and SGD variants such as momentum (Sutskever et al., 2013) and Adagrad (Duchi et al., 2011) have been used to achieve state of the
art performance. 

> 확률적 경사하강법은 깊은 신경망을 효율적으로 학습하는데에 증명되었으며
> 그리고 모멘텀, AdaGrad 등과 같은 SGD의 변형이 SOTA를 달성하였다.

SGD optimizes the parameters $\Theta$ of the network, so as to minimize the loss
> SGD 네트워크의 경사($\Theta$)를 최적화하며 loss 값을 줄인다.


$$ \Theta = argmin_\theta \frac{1}{N}\sum_{i=1}^N \ell(x_i, \Theta)$$

where x1...N is the training data set. 

> x1, ... N은 학습 데이터셋이다.

With SGD, the training proceeds in steps, and at each step we consider a minibatch x1...m of size m. 

> SGD를 사용함으로써, 학습은 순차적으로 진행되며 우리는 미니배치 x1 부터 m까지의 단계를 고려하낟.

The mini-batch is used to approximate the gradient of the loss function with respect to the parameters, by computing

> 미니배치는 손실함수를 최적의 경사에 근접하게 하기 위해 사용된다. 아래와 같은 연산을 통해서..

$$\frac{1}{m}\frac{\partial \ell(x_i, \Theta)}{\partial \Theta}$$


Using mini-batches of examples, as `opposed` to one example
at a time, is helpful in several ways.

> 미니배치를 사용하는 것은 다양한 방면에 도움을 준다.

First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. 

> 첫째, 미니 배치에 대한 손실의 기울기는 배치 크기가 증가함에 따라 품질이 향상되는 훈련 세트에 대한 기울기의 추정치이다.

Second, computation over a batch can be much more efficient than m computations for individual examples, due to the parallelism afforded by the modern computing platforms.

> 둘째, 미니배치를 사용하는 것은 최근 계산 플랫폼들이 병렬처리를 허용하기 때문에, 각각을 m번 계산하는것보다 더 효율적이다.

While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. 

> 반면, SGD는 단순하고 효율적이지만, 그것은 hyper parameter 튜닝을 정교하게 해야하며, 학습율또한 정규하게 작업해야한다. 또한 초기가중치도 그렇다.

The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as the network becomes deeper.

> 그 학습은 모든 input이 각각의 layer가 진행될대마다 영향을 끼치기 때문에, 깊어질수록 영향이 크다.

The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution. 

> 각 층의 레이어들은 새로운 분포에 대한 변화를 필요로 한다.

When the input distribution to a learning system changes, it is said to experience `covariate shift` (Shimodaira, 2000). 

> 학습 때, 입력값의 분포가 변한다면 covariate shift(공변량 이동)를 겪는다.

This is typically handled via `domain adaptation`(Jiang, 2008). 

> 이것은 도메인 적응(?)을 통해 해결된다.

However, the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network or a layer. 

> 그러나 공변량 이동의 개념은 학습 시스템 전체 뿐 아니라, 하위 네트워크 및 레이어에도 영향을 끼친다.

Consider a network computing
> 아래와 같은 network 연산을 고려한다.

$$\ell = F_2(F_1(u, \Theta_1), \Theta_2)$$

where F1 and F2 are arbitrary transformations, and the parameters  $\Theta_1, \Theta_2$ are to be learned so as to minimize the loss ℓ. 

> F1, F2는 임의의 변형값이며, 파라미터 $\Theta_1, \Theta_2$ 는 loss $\ell$ 을 최소화시키도록 학습한다.

Learning $\Theta_2$ can be viewed as if the inputs $x = F_1(u, \Theta_1)$ are fed into the sub-network
> $\Theta_2$를 학습시키는 것은 $x = F_1(u, \Theta_1)$를 input으로 하여, 하위 계층 네트워크에 입력되는것으로 볼 수 있다.

$$\ell = F_2(x, \Theta_2)$$

For example, a gradient descent step
> 경사 하강법 예시는 다음과 같다.

$$ \Theta_2 \leftarrow \Theta_2 - \frac{\alpha}{m} \sum_i^m \frac{\partial F_2(x_i), \Theta_2}{\partial \Theta_2} $$


(for batch size $m$ and learning rate α) is exactly equivalent to that for a `stand-alone network` $F_2$ with input $x$. 

> 배치사이즈 $m$과 학습률 $\alpha$는 정확히 stand-alone 네트워크인 $x$를 input으로 하는 $F_2$와 정확히 동일하다.

Therefore, the input distribution properties that make training more efficient – such as having the same distribution between the training and test data – apply to training the sub-network as well. 

> 따라서 input 데이터의 분포 특성은 학습을 더 효율적으로 만든다. such as)학습데이터와 테스트데이터의 분포가 같을 경우 - 하위 네트워크에도 잘 적용된다.

As such it is advantageous for the distribution of $x$ to remain fixed over time. 

> 따라서, $x$의 분포는 시간이 지나더라도 유지되는것이 유리하다. 

Then, $\Theta_2$ does not have to readjust to compensate for the change in the distribution of x.
> 그렇게된다면, $\Theta_2$는 $x$의 분포 조정을 위해 재조정 할 필요가 없다.

Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well.

> 서브 네트워크에 대한 입력의 고정된 분포는 하위 네트워크 밖의 계층들에게도 긍정적인 결과를 가져올 것이다.

Consider a layer with a sigmoid activation function $z = g(Wu + b)$ where $u$ is the layer input, the weight matrix $W$ and bias vector $b$ are the layer parameters to be learned, and $g(x) = \frac{1}{1+exp(−x)}$. As $|x|$ increases, $g′(x)$ tends to zero.

> 위와 같은 시그모이드 활성화함수를 고려하여, $u$가 레이어의 input임을 고려하면 , $W$와 $b$는 레이어의 파라미터에의해 학습될것이며, $|x|$ 가 높을수록 경사는 0에 근사하게 될 것이다.


This means that for all dimensions of $x = Wu+b$ except those with small absolute values, the gradient flowing down to $u$ will vanish and the model will train slowly.

> 이것은 모든 차원이 $x$의 절대값을 사용하면, 절대값이 작은 경우를 제외하고는 $u$는 소멸될 것이다.

However, since $x$ is affected by $W, b$ and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of $x$ into the `saturated regime` of the nonlinearity and slow down the convergence

> 그러나 x는 W, b와 아래의 모든 층의 파라미터에 영향을 받기 때문에 훈련 중에 x의 많은 차원을 비선형성의 포화상태로 이동시키고 수렴을 늦출 것이다.


This effect is amplified as the network depth increases. 

> 이러한 효과는 네트워크 깊이가 증가함에 따라 증폭됩니다.

In practice, the `saturation problem` and the resulting vanishing gradients are usually addressed by using `Rectified Linear Units` (Nair & Hinton, 2010) ReLU(x) = max(x, 0), careful initialization (Bengio & Glorot, 2010; Saxe et al., 2013), and small learning rates. 

> 경사 소멸 및 급격하게 경사가 변화는 문제는 ReLU 활성화함수를 통해 어느정도 해결이 되었었다. 또한 초기값을 다루거나 learning rate를 바꾸는 방식으로도.

If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

> 그러나 비선형성인 Input 이 좀더 안정적으로 네트워크에 들어와서 학습이 된다면, 포화상태에 빠질 가능성이 낮아지고, 훈련속도 또한 가속화될 것이다.

We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as `Internal Covariate Shift`. 

> 우리는 훈련 과정에서 심층 네트워크의 내부 노드 분포의 변화를 내부 공변량 이동이라고 한다.

Eliminating it offers a promise of faster training. 

> 이것을 제거함으로써 학습속도가 빨라진다.

<font color='red'>We propose a new mechanism, which we call Batch Normalization, that takes a step towards reducing internal covariate shift, and in doing so dramatically accelerates the training of deep neural nets.  </font>

> 우리는 내부 공변량 이동을 줄이고, 그렇게 함으로써 심층 신경망의 훈련을 획기적으로 가속화하는 배치 정규화라고 불리는 새로운 메커니즘을 제안한다.


It accomplishes this via a normalization step that fixes the means and variances of layer inputs. 

> 이것은 계층 입력의 평균과 분산을 수정하는 정규화 단계를 통해 달성된다.

Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values.

> 배치 정규화는 또한 매개 변수의 척도 또는 초기 값에 대한 그레이디언트의 의존성을 줄임으로써 네트워크를 통한 그레이디언트 흐름에 유익한 영향을 미친다.

This allows us to use much higher learning rates without the risk of divergence. 

> 이것은 학습율을 높여도, 경사가 발산하는 문제가 없도록 한다.

Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014).

> 게다가, 배치정규화는 모델에 규제를 해주며, DropOut의 필요성을 줄여준다.

Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

> 마지막으로, 배치 정규화를 사용하면 비선형적인 활성화함수를 쓸 수 있다??


In Sec. 4.2, we apply Batch Normalization to the bestperforming ImageNet classification network, and show that we can match its performance using only 7% of the training steps, and can further exceed its accuracy by a `substantial margin`

> 4.2절에서는 Batch Normalization을 가장 성능이 좋은 ImageNet 분류 네트워크에 적용하고 학습 단계의 7%만 사용하여 동일한 성능을 낼 수 있으며 정확도도 훨씬 능가할 수 있음을 보여준다.

Using an ensemble of such networks trained with Batch Normalization, we achieve the top-5 error rate that improves upon the best known results on ImageNet classification.

> 배치정규화 앙상블을 적용하여, 우리는 ImageNet 분류기의 성능을 향상시켰다.
> : top-5 검증데이터셋의 error가 4.9%로 인간이 평가하는 것을 뛰어넘었다.

## 2. Towards Reducing Internal Covariate Shift

We define Internal Covariate Shift as the change in the distribution of network activations due to the change in network parameters during training. 

> 우리는 내부 공변량 이동을 훈련 중 네트워크 매개 변수의 변화로 인한 네트워크 활성화 분포의 변화로 정의한다.

To improve the training, we seek to reduce the internal covariate shift.

> 훈련을 개선하기 위하여,  우리는 내부 공변량 이동을 줄이는 방법을 탐색한다.

By fixing the distribution of the layer inputs x as the training progresses, we expect to improve the training speed.

> 학습이 진행됨에 따라 레이어 입력 x의 분포를 고정함으로써 학습 속도를 향상시킬 수 있을 것으로 기대하고 있습니다.

It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are `whitened` – i.e., linearly transformed to have zero means and unit variances, and `decorrelated`. 

> 입력데이터가 whitend(? : 아마 표준화같은개념) 수렴속도가 더 빨라진다는 것은 이미 잘 알려져있다. - 평균 0을 갖으며, 단위 분산과 비상관성을 갖는다. 

As each layer observes the inputs produced by the layers below, it would be advantageous to achieve the same whitening of the inputs of each layer.

> 각 층이 아래 층에 의해 생성된 입력을 관찰하기 때문에 각 층의 입력에 대해 동일한 화이트닝을 달성하는 것이 유리할 것이다.

By whitening the inputs to each layer, we would take a step towards achieving the fixed distributions of inputs that would remove the ill effects of the internal covariate shift.

> 각 층에 대한 입력을 whitening 함으로써, 우리는 내부 공변량 이동의 부작용을 제거하기 위해,  입력의 고정된 분포를 위해 단계를 진행할것이다.


We could consider whitening activations at every training step or at some interval, either by __modifying the
network directly__ or by __changing the parameters of the optimization algorithm__ to depend on the `network activation values` (Wiesler et al., 2014; Raiko et al., 2012; Povey et al., 2014; Desjardins & Kavukcuoglu).

> 우리는 network를 직접 수정하거나, 파라미터 최적화 알고리즘을 변경함으로써 각 단계별 whitening 활성화를 진행할 수 있다. 
> 이는 network activation value에 의존적이다. (활성화함수 말하는건지?)

However, if these modifications are `interspersed` with the optimization steps, then the gradient descent step may attempt to update the parameters in a way that requires the normalization to be updated, which reduces the effect
of the gradient step

> 그러나 이 경우 드문드문 최적화 단계가 있다면, 경사 하강 단계는 정규화를 업데이트해야 하는 방식으로 매개변수를 업데이트하려고 시도할 수 있으며, 이는 경사 단계의 영향을 감소시킨다.

For example, consider a layer with the __input $u$__ that adds the learned bias $b$, and normalizes the result by subtracting the mean of the activation computed over the training data: $\hat{x} = x − E[x]$ where $x = u + b$, $X = \{x1...N\}$ is the set of values of $x$ over the training set, and $E[x] = \frac{1}{N}\sum_i^N xi$.

> 단계별로 input data에 대해서 정규화 과정을 진행하는것으로 해석됨.

If a gradient descent step ignores the dependence of $E[x]$ on $b$, then it will update $b \leftarrow b + \triangle b$, where  $\triangle b \propto −∂ℓ/∂ \hat{x}$
> 만약 경사하강법이 기대값$E[x]$ 와 $bias$를 무기한다면 위와 같이 업데이트 될 것이다.

Then $u+(b+\triangle b)-E[u+(b+\triangle b)]=u+b-E[u+b]$.
> 위와 같이 업데이트가 됐을 것이다?

Thus, the combination of the update to b and subsequent change in normalization led to no change in the output of the layer nor, consequently, the loss. 

> 그러므로, $b$에 대한 업데이트와 그에 따른 정규화 변경의 조합은 계층 출력의 변화도, 결과적으로 손실도 초래하지 않았다.

As the training continues, $b$ will grow indefinitely while the loss remains fixed.

> 훈련이 계속될수록 $b$는 손실이 고정된 상태에서 무한 성장한다.

This problemcan get worse if the normalization not only centers but also scales the activations.
> 이 문제는 정규화가 중심일 뿐만 아니라 활성화 함수로 스케일링 한다면, 더 악화될 수 있다.

We have observed this empirically in initial experiments, where the model blows up when the normalization parameters are computed outside the gradient descent step.

> 정규화 매개 변수가 경사 하강 단계 밖에서 계산될 때 모델이 폭파되는 초기 실험에서 이를 경험적으로 관찰했다.

The issue with the above approach is that the gradient descent optimization does not take into account the fact that the normalization takes place. 

> ㅋㅋ

<font color='red'>To address this issue, we would like to ensure that, for any parameter values, the network always produces activations with the desired distribution.  </font>

> 이 문제를 해결하기 위해, 우리는 모든 매개 변수 값에 대해 네트워크가 항상 원하는 분포로 활성화를 생성하도록 보장하고자 한다.


Doing so would allow the gradient of the loss with respect to the model parameters to account for the normalization, and for its dependence on the model parameters $\Theta$.

> 그렇게 하는 것은 정규화와 모델 매개변수 $\Theta$에 대한 의존성을 설명하기 위해 모델 매개변수에 대한 손실의 기울기를 허용할 것이다.

Let again x be a layer input, treated as a vector, and X be the set of these inputs over the training data set. 

> 다시 x를 벡터로 취급하는 레이어 입력으로 하고 X를 훈련 데이터 집합에 대한 이러한 입력들의 집합으로 하자.

The normalization can then be written as a transformation
> 그 정규화의 변환은 아래식처럼 쓸 수 있다.

$$\hat{x} = Norm(x, \chi)$$

which depends not only on the given training example $x$ but on all examples $\chi$ – each of which depends on $\Theta$ if $x$ is generated by another layer. For backpropagation, we would need to compute the `Jacobians`

> 이는 주어진 학습 예제 𝑥 뿐만 아니라 모든 예제 𝜒에 따라 달라지는데, 각 예제 𝜒는 $x$가 다른 계층에 의해 생성될 경우 Θ에 따라 달라진다. 
> 역전파를 위해서는 자코비안들을 계산한다.

$$ \frac{\partial Norm(x, \chi)}{\partial x} and \frac{\partial Norm(x, \chi)}{\partial \chi} ;$$

ignoring the latter term would lead to the explosion described above. 
> 후자의 용어를 무시하면, 위에서 설명한 기울기 폭발이 일어날 것이다.

Within this framework, whitening the layer inputs is expensive, as it requires computing the `covariance matrix` $Cov[x] = Ex_{∈\chi} [xxT ] − E[x]E[x]^T$ and its inverse square root, to produce the `whitened activations` $Cov[x]^{-1/2}(x − E[x])$, as well as the derivatives of these transforms for backpropagation. 

> 이 프레임워크에서는, layer input을 whkitening하는 연산은 비싸다. 우리는 연산을 위해 공분산행렬을 필요로 하며, 그것은 역수 및 루트를 씌워서 `whitend activations`를 만든다. 우리는 역전파를 위해 이것을 파생한다.

This motivates us to seek an alternative that performs input normalization in a way that is differentiable and does not require the analysis of the entire training set after every parameter update.

> 이것은 우리가 차별화 가능하고 모든 매개 변수 업데이트 후 전체 학습데이터의 분석을 요구하지 않는 방식으로 입력 정규화를 수행하는 대안을 찾도록 만든다.

Some of the previous approaches (e.g. (Lyu & Simoncelli, 2008)) use statistics computed over a single training example, or, in the case of image networks, over different feature maps at a given location.

> 이전 접근법 중 일부는 단일 훈련 예시에 대해, 또는 이미지 네트워크의 경우 주어진 위치의 다른 특징맵에 대해 계산된 통계값 사용한다.

However, this changes the representation ability of a network by discarding the absolute scale of activations.
> 그러나 이는 활성화의 절대적인 scale을 버림으로써 네트워크의 표현정보를 변화시킨다.

We want to a preserve the information in the network, by normalizing the activations in a training example relative to the statistics of the entire training data.
> 우리는 전체 훈련 데이터의 통계를 기준으로 훈련 예제의 활성화를 정규화하여 네트워크의 정보를 보존하고자 한다.

##  3. Normalization via Mini-Batch Statistics

Since the full whitening of each layer’s inputs is costly and not everywhere differentiable, we make two necessary simplifications. 

> 각 층의 입력에 대한 완전한 화이트닝은 비용이 많이 들고 어디에서나 구별할 수 있는 것은 아니기 때문에, 우리는 두 가지 필요한 단순화를 한다.

The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1. 

> 첫 번째는 레이어 입력과 출력의 특징을 공동으로 화이트닝하는 대신, 우리는 각 스칼라 특징을 0과 1의 분산으로 만들어 독립적으로 정규화할 것이다.

For a layer with $d$-dimensional input $x = (x^{(1)} . . . x^{(d)})$, we will normalize each dimension

> 위와같은 $d$ 차원의 input이 들어온다면, 우리는 아래와같이 각각의 차원은 정규화시킬것이다.

$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}] }{\sqrt{Var[x^{(k)}]}}$$


where the expectation and variance are computed over the training data set. 

> 여기서 학습 데이터 세트에 대한 기대값과 분산을 계산합니다.

As shown in (LeCun et al., 1998b), such normalization speeds up convergence, even when the features are not decorrelated.

> (LeCun et al., 1998b)에서 볼 수 있듯이, 이러한 정규화는 특징이 상관관계가 없는 경우에도 수렴 속도를 높인다.

Note that simply normalizing each input of a layer may change what the layer can represent. 

> 단순히 계층의 각 입력을 정규화하는 것은 계층이 나타낼 수 있는 것을 변화시킬 수 있다는 것을 주목하라.

For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. 

> 예를 들어, 시그모이드의 입력을 정규화하면 비선형성의 선형으로 제한된다.

To address this, we make sure that the transformation inserted in the network can represent the `identity transform`.
> 이를 해결하기 위해 네트워크에 삽입된 변환이 identity transform(? : 기본 정보로 추측) 을 나타낼 수 있는지 확인한다.

To accomplish this, we introduce, for each activation $x^{(k)}$, a pair of parameters $γ^{(k)}, β^{(k)}$, which scale and shift the normalized value:

> 각각의 모든 활성화함수 $x^{(k)}$에 대해, 각각의 쌍인 $γ^{(k)}, β^{(k)}$,는 스케일링 및 정규화값을 shift한다.

These parameters are learned along with the original model parameters, and restore the representation power
of the network. 

> 이러한 매개 변수는 원래 모델 매개 변수와 함께 학습되고 네트워크의 표현 능력을 복원한다.

Indeed, by setting $γ^{(k)} = \sqrt{Var[x(k)]}$ and $β^{(k)} = E[x^{(k)}]$, we could recover the original activations,
if that were the optimal thing to do.

> $γ^{(k)} = \sqrt{Var[x(k)]}$ and $β^{(k)} = E[x^{(k)}]$와 같이 설정한다면, 우리는 기본 활성화를 복원할 수 있다.


In the batch setting where each training step is based on the entire training set, we would use the whole set to normalize activations.

> 각 훈련 단계가 전체 훈련 세트를 기반으로 하는 배치 설정에서 전체 세트를 사용하여 활성화를 정규화한다.

However, this is impractical when using stochastic optimization. 

> 그러나 이것은 확률적 최적화를 사용할 때 비현실적이다.

Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each activation.

> 따라서, 우리는 두 번째 단순화를 한다 : 확률적 경사 훈련에서 미니 배치를 사용하기 때문에, 각 미니 배치는 각 활성화의 평균과 분산에 대한 추정치를 생성한다.

This way, the statistics used for normalization can fully participate in the gradient backpropagation. 

> 이러한 방식으로 정규화에 사용되는 통계는 그레이디언트 역전파에 완전히 참여할 수 있다.

Note that the use of minibatches is enabled by __computation of per-dimension variances__ rather than joint covariances; 

> 미니배치의 사용은 전체 공분산보다는 차원당 분산 계산에 의해 가능하다.

in the joint case, regularization would be required since the mini-batch size is likely to be smaller than the number of activations being whitened, resulting in singular covariance matrices.

> 이 전체 경우, 미니 진동 크기가 흰색으로 변하는 활성의 수보다 작을 가능성이 높기 때문에 정규화가 요구되어 단수 공분산 행렬이 생성된다.

Consider a mini-batch B of size m.

> 크기가 m인 미니배치 B를 고려해보자. 

Since the normalization is applied to each activation independently, let us focus on a particular activation $x^{(k)}$ and omit $k$ for clarity. 

>정규화는 각 활성화에 독립적으로 적용되므로 특정 활성화 $x^{(k)}$에 초점을 맞추고 명확성을 위해 $k$를 생략합시다.

We have $m$ values of this activation in the mini-batch,

>우리는 미니배치안에서 $m$ 값의 활성화를 갖는다.

$$ B = \{{x_{1...m}}\}.$$

Let the normalized values be $\hat{x}_{1...m}$, and their linear transformations be $y_{1...m}$. 

> $\hat{x}_{1...m}$ 값을 정규화 하고, $y_{1...m}$ 로 선형변환 하자.

We refer to the transform as the <font color='red'>Batch Normalizing Transform</font>. 

> 아래와 같은 Batch Normalizing Transform 을 볼 수 있다.

$$BN_{\gamma,\beta} : x_{1...m} \rightarrow y_{1...m}$$

We present the BN Transform in Algorithm1. 

> 우리는 Algoritm1에 변환과정을 표현한다.

In the algorithm, $\epsilon$ is a constant added to the mini-batch variance for `numerical stability`.

> 해당 알고리즘에서 $\epsilon$는 수치의 안정성을 위해 미니 배치 분산에 추가되는 상수이다.

![algorithm1](../pics/bn_algorithm1.png)

The BN transform can be added to a network to manipulate any activation.