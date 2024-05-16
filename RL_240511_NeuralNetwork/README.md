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

```python
def forward(self, x):
    self.hidden_layers = [
        nn.Linear(16,16)
        for _ in range(3)
    ]
    return x
```

또 layer를 만드는 과정은 위와 같이도 짤 수 있다.

```python
def forward(self, x):
    x = self.input_layer(x)
    x = self.activation(x)
    x = self.hidden_layer(x)
    x = self.activation(x)
    x = self.output_layer(x)
```

???????

#### 딥러닝 진행행

```python
network = NeuralNetwork()
loss_function = nn.MSELoss() # 손실함수 선언

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
```

Adam?

```python
optimizer.zero_grad()
output = network(x)
loss = loss_function(output,y)
loss.backward()
optimizer.step()
```

- optimizer.zero_grad() : 경사하강법 처음부터
- output = network(x) : 레이어를 다라 진행한 결과
- loss = loss_function(output,y) : 손실함수로 오차 계산
- loss.backward() : 역전파
- optimizer.step() : 경사하강법 한번 진행

위 과정이 인공지능이 한번 학습하는 과정이며 원하는 횟수만큼 반복하여 학습을 진행하면 된다.




```python
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the network and the loss function.
network = NeuralNetwork()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

losses = []
pbar = tqdm(range(100000), desc="Loss: --")
for epoch in pbar:
    x = torch.Tensor([torch.randn(1)])
    y = x**3 + x**2 - x - 1
    optimizer.zero_grad()
    output = network(x)
    loss = loss_function(output,y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        pbar.set_description(f"Loss: {loss.item():.3f}")
```
위 코드는 $y=x^{3}+x^{2}-x-1$를 딥러닝으로 근사하는 코드이다.

학습결과를 출력하여 보면 아래와 같다.

![image](https://github.com/nkmin0/2024_RL/assets/162765658/8ed39c05-3963-42d8-acff-625313efcacc)

그런데 그래프의 xlim을 바꾸어 보면

![image](https://github.com/nkmin0/2024_RL/assets/162765658/7ca52239-c254-4d42-bb78-54300af2e3e8)

위와같이 전체 범위에 대해서는 근사가 제대로 되어지지 않았다. 이는 지도학습의 단점으로 학습한 지역의 데이터에서는 높은 정확도를 보이지만 학습하지 않은 부분에 대해서는 근사하지 못한다는 것이다.
