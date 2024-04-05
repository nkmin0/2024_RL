# 2.5차시 2024.04.?

![image](https://github.com/nkmin0/2024_RL/assets/162765658/516ff8da-273c-4455-8493-c7b2ddbdea5c)

## 환경 생성

```python
class MyEnv(gym.Env): #환경
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(4, start=0)
        # 0 : 수업
        # 1 : 야자
        # 2 : 집
        # 3 : 시험

        self.action_space = gym.spaces.Discrete(3)


    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        q = random.random()
        next_state = self.state
        reward = 0
        done = False

        if self.state == 0: # 수업
            if action == 0: #딴짓
                if q < 0.2:
                    next_state = self.state + 1
                reward = -1
            else: # 공부
                next_state = self.state + 1
                reward = -2

        elif self.state == 1: # 야자
            if action == 0: # 떙땡이
                if q < 0.9:
                    next_state = self.state - 1
                else:
                    next_state = self.state + 1
                reward = 1
            else: # 공부
                next_state = self.state + 1
                reward = -2

        elif self.state == 2: # 집
            if action == 0: # 유튜브
                reward = -1
            elif action == 1: # 벼락치기
                reward = 5
                next_state = self.state + 1
            else: # 잠
                next_state = self.state + 1

        elif self.state == 3: # 시험 결과 나옴
            done = True

        else:
            done = True
            #print("error")

        self.state = next_state  # 다음 상태로 업데이트

        return next_state, reward, done, {}

```

## 정책 적용 및 학습

```python
class Mypolicy():
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.1 # 학습률
        self.discount_factor = 0.9 # 할인율
        self.exploration_rate = 0.5  # 탐험 확률
        self.exploration_decay = 0.99 # 탐험 확률 조정
        self.min_exploration_rate = 0.01 # 최소 탐험 확률
        self.Q = {}
        self.rewards = []
        self.choose_what = [[0,0,0],[0,0,0],[0,0,0]]
        self.x=0
        self.state = self.env.reset()


        self.Q[(0, 0)] = 0.0
        self.Q[(0, 1)] = 0.0
        self.Q[(1, 0)] = 0.0
        self.Q[(1, 1)] = 0.0
        self.Q[(2, 0)] = 0.0
        self.Q[(2, 1)] = 0.0
        self.Q[(2, 2)] = 0.0

```

```python
def choose_action(self):
    if self.state == 3: # state가 3일 때는 의미 없음
        return 0
    elif random.uniform(0, 1) < self.exploration_rate: # 일정 확률로 탐험
        if self.state < 2:
            return random.randint(0, 1)
        else:
            return self.env.action_space.sample()  # 탐험
    else: # 그리디
        if self.state < 2:
            if self.Q[(self.state,0)]>=self.Q[(self.state,1)]:
                return 0
            else:
                return 1
        else:
            if self.Q[(self.state,1)]>=self.Q[(self.state,0)] and self.Q[(self.state,1)]>=self.Q[(self.state,2)]:
                return 1
            elif self.Q[(self.state,0)]>=self.Q[(self.state,1)] and self.Q[(self.state,0)]>=self.Q[(self.state,2)]:
                return 0
            else:
                return 2
```

```python
def update_Q(self, state, action, next_state, reward):
    current_q_value = self.Q.get((state, action), 0)
    max_future_q_value = max(self.Q.get((next_state, a), 0) for a in range(self.env.action_space.n))
    # 가장 가치가 높은 것을 찾음
    new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q_value)
    # 값 업데이트

    self.Q[(state, action)] = new_q_value
```
### Q Update

$$ Q_{\pi}(s,a) = E_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \cdots | S_{T}=s, A_{t} = a] $$

어떤 State(S)에서 어떤 Action(A)를 했을 때 그 행동의 가치를 계산한다. 이를 행동 가치 함수라고도 부르고 discount factor($\gamma$)를 사용하여 특정 action을 하였을 때 reward의 총합의 예측값을 구한다.

discount factor($\gamma$)는 현재 얻는 보상이 미래에 얻는 보상보다 얼마나 더 중요한지를 나타내는 값이다.

현재 상태로 부터 $t$초가 흐른 후에 얻는 보상은 $\gamma^{t}$ 만큼 할인 되어 계산 한다. 따라서 이를 정리하면 

![image](https://github.com/nkmin0/2024_RL/assets/162765658/3402ad4a-2b10-46e1-b93e-f5dafe7f77c9)

```python
def train(self, training_steps=100):
    for i in range(training_steps):
        self.state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action()
            if self.state != 3:
                self.choose_what[self.state][action]+=1
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            self.update_Q(self.state, action, next_state, reward)

            self.state = next_state

            if self.state == -1:
                print("err")

        # 학습률 업데이트
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

        self.rewards.append(total_reward)

        if i%10 == 0:
            print("i:", i, "Total Reward:", total_reward)

    print("Training finished.")
```

```python
env = MyEnv()

agent = Mypolicy(env)
agent.train()
```

## 학습 결과

![image](https://github.com/nkmin0/2024_RL/assets/162765658/11fcf901-e00a-4cac-99d9-c9223a47fcba)

학습 결과 딴짓 땡땡이같이 공부하는 것 보다 보상적인 측면에서 이득을 볼 수 있지만 확률이 낮다. 그래서

공부 --> 공부 --> 벼락치기 를 하면서 r=1인 경우가 가장 많이 나왔다.

![image](https://github.com/nkmin0/2024_RL/assets/162765658/8a0f067a-a0eb-40d2-9bb8-ca35f6982d0f)
![image](https://github.com/nkmin0/2024_RL/assets/162765658/f288953c-fdf9-4fc1-8acd-11598a31ee82)
![image](https://github.com/nkmin0/2024_RL/assets/162765658/54f24842-3762-4a19-92c5-7b57bbc8b113)

![image](https://github.com/nkmin0/2024_RL/assets/162765658/e63485ab-57da-4897-abf1-07541b7a1872)

위 차트를 보면 상태가 0,1 일 때초반에는 딴짓, 땡땡이를 하지만 나중 갈수록 공부만 선택하는 것을 볼 수 있다.

또한 상태가 2일 때 벼락치기를 제외한 다른 선택지들이 모두 보상적인 측면에서 손해를 보기 때문에 벼락치기를 선택하는 비율이 매우 높은 것을 볼 수 있다.

