# 5차시 2024.05.11

## 인공신경망과 딥러닝

인공신경망이란?

![image](https://github.com/nkmin0/2024_RL/assets/162765658/86d99603-fa69-4920-b1b0-ec27846a1055)

뉴런의 구조를 단순화하여 만든 것. 생물학적인 신경 세포를 단순화 하여 모델링한 뉴런. 위 오른쪽 그림처럼 여러 신호를 받아 하나의 신호를 만들어 전달하는 역할을 한다.

![image](https://github.com/nkmin0/2024_RL/assets/162765658/121bcfdf-86cf-412d-a2bb-7d2e5527aaed)

이처럼 input layer와 output layer사이 여러개의 hidden layer가 있는 구조를 딥러닝이라고 한다.

## 인공신경망을 이용한 지도학습

지도학습은 문제와 정답이 주어졌을 때 정답을 가장 잘 표현하는 함수를 찾는 과정이다. 

...

### 보편근사정리

- 하나의 은닉층을 가지는 인공신경망은 임의의 연속인 함수를 원하는 정도의 정확도로 근사할 수 있다.
- 즉 인공신경망을 사용하면 모든함수를 거의 정확하게 표현할 수 있다.


### $y=x^{3}+x^{2}-x-1$ 근사하기

#### pytorch 라이브러리 불러오기
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

#### 딥러닝 모델 구축

```python
class NeuralNetwork(nn.Module): # 모듈 상속

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # layer 선언
        self.input_layer = nn.Linear(1,16) # layer 1
        self.activation = nn.ReLU() # 활성화 함수 ex) ReLU, sigmoid ...
        self.hidden_layer1 = nn.Linear(16,1024) 
        self.hidden_layer2 = nn.Linear(1024,16)
        self.output_layer = nn.Linear(16,1) # layer 2

    def forward(self, x): 
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.hidden_layer2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x        
```

![image](https://github.com/nkmin0/2024_RL/assets/162765658/8b04c86f-929f-41ad-abc5-79687d1e4569)

위 사진처럼 구성한 모델은 $x$에 $w$(가중치)를 곱하고 $b$(편향)을 더하는 과정을 거친다.

이 과정은 결국 일차함수이기 때문에 

```python
self.input_layer = nn.Linear(1,16)
```

처럼 선형으로 코드를 짤 수 있다.

또 layer를 만드는 과정은 아래와 같이도 짤 수 있다.

```python
def forward(self, x):
    self.hidden_layers = [
        nn.Linear(16,16)
        for _ in range(3)
    ]
    return x
```



