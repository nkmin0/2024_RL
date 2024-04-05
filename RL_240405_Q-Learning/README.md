# 2.5차시 2024.04.?

![image](https://github.com/nkmin0/2024_RL/assets/162765658/516ff8da-273c-4455-8493-c7b2ddbdea5c)

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
        state = 0
        return state

    def step(self, state, action):
        q=random.random()
        next_state = state
        reward = 0
        done = False

        if state == 0: # 수업
            if action == 0: # 땡땡이
                if q<0.2:
                    next_state = state + 1
                reward = -1
            else: # 공부
                next_state = state + 1
                reward = -2

        elif state == 1: # 야자
            if action == 0: # 딴짓
                if q<0.9:
                    next_state = state - 1
                else:
                    next_state = state + 1
                reward = 1
            else: # 공부
                next_state = state + 1
                reward = -2

        elif state == 2: # 집
            if action == 0: # 유튜브
                reward = -1
            elif action == 1: # 벼락치기
                reward = 5
                next_state = state + 1
            else: # 잠
                next_state = state + 1

        elif state == 3: # 시험 결과 나옴
            done=True

        else:
            done=True
            #print("error")

        return next_state, reward, done, {}
```
