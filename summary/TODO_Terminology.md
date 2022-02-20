### Vanishing gradient / Exploding gradient

기울기 소멸 문제(vanishing gradient problem)는 신경망의 활성함수의 도함수값(gradient)이 계속 곱해지다보면 가중치에 따른 결과값의 기울기가 0이 되어 버려서, 경사 하강법을 이용할 수 없게 되는 문제입니다. 반대로 기울기값이 계속 증폭될 경우 기울기 폭발 문제(exploding gradient problem)가 발생합니다.

출처 : https://ko.wikipedia.org/wiki/%EA%B8%B0%EC%9A%B8%EA%B8%B0_%EC%86%8C%EB%A9%B8_%EB%AC%B8%EC%A0%9C

### Identity mapping

Identity mapping에서 `Identity`는 자기자신을 의미합니다. 수학에서 Identity function(항등 함수)의 Identity와 동일한 의미를 갖습니다. Identity mapping은 자기자신, 즉 인풋값을 그대로 매핑하는 역할을 합니다.

### top-1 / top-5 error

top-1 / top-5 error는 image classification 분야에서 사용되는 metric입니다.

모델이 가장 높은 확률로 예측한 결과가 정답값과 동일하다면 top-1 에러는 0이 될 것입니다. Top-5 에러는 모델이 예측한 결과 중 확률이 가장 높은 5개 중 정답값과 동일한 예측값이 있는지를 기준으로 측정합니다.