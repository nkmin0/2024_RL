# Selling Bread

## 1. 서론

어떠한 물건을 판매하는 입장에서 그 물건의 수요와 공급은 매우 중요하다. 그 물건의 수요량에 맞게 공급량을 조절하면 효율적인 판매를 할 수 있기 때문이다. 특히 빵집같이 미리 만들어놓은 제품을 무제한으로 진열할 수 없이 일정 시간이 지나가면 버려야 하는 경우에는 이런 공급량의 조절이 매출에 큰영향을 끼칠것 이다.

빵집을 예로 들었을 때

- 수요보다 빵을 많이 만든 경우 : 팔리지 않은 빵을 버려야 하기 때문에 그만큼 손해를 보게 된다.
- 수요보다 빵을 적게 만든 경우 : 고객이 원하는 빵이 가게에 없기 때문에 그 손님이 다시 가게를 방문하지 않을 가능성이 높아져서 잠재적인 매출이 감소하게 된다.

따라서 앞서 말했듯이 빵집에서 공급을 조절하는 것은 매우 중요하다. 하지만 수요가 정해져 있지 않고 요일, 날씨 등 다양한 요인들 때문에 매번 정확하게 빵을 만들 수 없다. 따라서 매출의 손실을 최소화 하기 위해 이를 강화학습으로 해결해보고자 한다.

<!--
## 이론적 배경

## 선행연구 

## 연구방법 및 절차
-->

### 문제 추상화

실제 상황에 대해서 문제상황을 해결하기 위해 먼저 문제를 추상화 하여 해결하고 점차 변수를 증가해 나가면서 실제 상황에서도 적용해보고자 한다.

가장 먼저 특정한 빵 $A$의 수요와 공급에 대한 문제 상황을 먼저 해결하려한다. 
빵 $A$가 하루 평균 $m$개가 팔리는데 확률적인 요인으로 $m$개 보다 더 팔리거나 더 적게 팔린다고 가정한다. 학습을 시작할 때에는 하루에 몇개의 $A$가 팔리는지 모르는 상황에서 시작한다. 

오늘 몇개의 빵을 만들지($N$)를 행동을 통해 결정할 수 있고 이후 오늘 빵의 수요인 $m + \alpha$개에 의해 에이전트가 받는 보상을 결정한다. 환경에 상태에는 현재까지 빵 $A$가 하루 평균 몇개가 팔렸는지($M$)와 오늘이 영업 개시 이후 몇일차인지($t$)를 저장한다. 이 상태에서는 다른 변수가 없으므로 손해를 최소화 하기 위해 수요와 공급의 차 만큼 패널티를 주어 학습을 하려 한다. 

$$ R_{t} = -| N - (m + \alpha_{t}) | $$

이를 통해서 환경을 만들면

```python
class MyEnv(gym.Env):
    def __init__(self, learning_day=101, MAX_bread = 20):
        self.observation_space = gym.spaces.Discrete(learning_day, start=0) # 학습 날짜
        self.action_space = gym.spaces.Discrete(MAX_bread) # 최대 빵 개수

        self.m = 10 # 평균적으로 팔리는 빵의 개수
    
    def reset(self):
        self.day = 0
        return self.day
      
    def step(self, daily_bread):
        a = np.random.normal() * 2 / 1
        self.day+=1 # step함수 호출마다 하루씩 증가
        next_state = self.day
        done = False

        reward = -abs(daily_bread-(self.m+a)) # 보상함수 

        if self.day>=100:
            done = True

        return next_state, reward, done, {}
```

### 추가 해야할 변수

- 빵이 남아도 다음날 까지는 팔 수 있음 : 빵의 유통기한을 정해서 빵이 남아도 다음날에는 팔 수 있도록 변수에 값을 저장해야함
- 요일마다 빵의 수요가 바뀜 : contextual bandit문제와 비슷함
- 날씨에 따라 빵의 수요가 바뀜 : 비가옴, 햇빛이 셈 등의 상황에 의해 수요가 일정 비율 감소함

이러한 변수를 추가해서 실제 상황에서도 강화학습을 통해 문제를 해결할 수 있도록 할 것이다.

- 만약 데이터가 있다면 : 그 데이터를 활용하여 시뮬레이션 환경 제작 및 학습 진행
- 만약 데이터가 없다면 : 직접 시뮬레이션 환경을 만들어야함
- 또는 논문이나 다른 글들에서 특정한 물건의 판매량이 평균을 기준으로 얼마만큼에 오차를 가지고 팔린다 라는 정보를 찾아 그걸 바탕으로 시뮬레이션 환경 제작

<!--
## 연구 결과

## 기대 성과
-->
