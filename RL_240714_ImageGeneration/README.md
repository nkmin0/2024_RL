# 2024.07.14

## 언어 모델

### 언어 토큰화

데이터가 전처리 되지 않은 상태라면, 데이터를 토큰화해야 한다. 컴퓨터에게 인식시키기 위해 쪼갠 언어의 최소 단위를 '토큰'이라 한다.

```python
from transformers import AutoTokenizer

sample_text = '언어 모델의 이해'
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

tokens = tokenizer.tokenize(sample_text)
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_to_id_mapping = dict(zip(tokens, token_ids))
print(token_to_id_mapping)
```

출력

```python
['언어', '모델', '##의', '이해']
{'언어': 18010, '모델': 16505, '##의': 4042, '이해': 8226}
```

이와 같이 '언어 모델의 이해'라는 문장에서 각각을 토큰화 하여 숫자를 부여한다. 

- tokenizer : 토큰으로 변환하는 모델
- tokens : 변환된 모델

### 언어 처리 모델

토큰화를 통해 컴퓨터가 이해할 수 있는 숫자 형태로 변환

![image](https://github.com/user-attachments/assets/3362cad1-5d96-4eed-8f92-cd1c1b4881b5)

문장의 길이가 일정하지 않으므로 전에 배웠던 바로 위 그림처럼 적용하기 힘들다. 

![image](https://github.com/user-attachments/assets/ddba44e7-75d4-4c58-bdc1-165695688ad7)

따라서 RNN을 사용해 입력한 문장을 토큰화 한 후 순차적으로 한개씩 모델에 적용되도록 한다.

## 이미지 생성 모델

### 인코더-디코더 구조

![image](https://github.com/user-attachments/assets/49ffd8d8-1c62-4c28-a3b5-c2ea2ff47d60)

- 인코더 : 주어진 입력값을 함축적인 정보로 표현하는 모델
- 디코더 함축된 정보를 다시 복원하는 모델

![image](https://github.com/user-attachments/assets/30163cec-c70e-494b-b8c0-f794edb67eb2)

이중 생성모델은 디코더 구조와 비슷함 

### Diffusion

![image](https://github.com/user-attachments/assets/63bf1333-6308-4d61-b0a1-860520ec0624)


## 인간 피드백을 이용한 강화학습

이미지 생성 분야에서 강화학습을 하고자 할 때 보상함수가 필요하다. 그런데 이런 보상함수를 사람이 생성된 이미지를 보고 "잘만들어졌다." / "잘 안만들어졌다." 를 직접 판단해 강화학습을 하기도 한다. 

![image](https://github.com/user-attachments/assets/4e82a3b6-67d7-41ec-8edd-3825d56caa22)

이미지 분야에서 생성된 이미지가 좋고 나쁨을 판단할 때 한개의 이미지 만으로는 프롬프트에 맞춰 잘 만들어 졌는지 확인할 수 없다. 따라서 보통 두개의 이미지를 생성한 후 둘 중 어떤 사진이 더 잘만들어졌는지를 판단해 이를 강화학습에 활용한다.

혹은 LLM을 통해 보상함수를 사용하기도 한다.

