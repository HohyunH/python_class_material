# python_class_material

본 저장소는 [2021 1st semester 데이터처리언어] 수업 실습자료 입니다.

아래와 같이 구성되어 있습니다.

1. Deep Learning Library(Keras of Pytorch) 기초
2. Crawling 기초
3. colab / github tutorial

## 1. Pytorch & Keras

#### 1-1 MLP Regressor

- sklearn에서 제공하는 보스턴 주택 가격 데이터 셋을 활용합니다.

```python
from sklearn.datasets import load_boston
bos = load_boston() 
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names 
df['Price'] = bos.target

```
- Multi Layer Perceptron(MLP)을 활용한 regression 문제를 해결하는 문제입니다.
- Loss Function으로 MSE loss를 활용합니다.
- Metric으로는 MSE score와 RMSE score로 성능을 측정합니다.

#### 1-2 CNN Image Classification

- torchvision에서 제공하는 CIFAR 10 데이터셋을 활용합니다.

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True) 

```

- Convolution Neural Network(CNN)을 활용한 Image Classification 문제를 해결하는 문제입니다.
- Loss Function으로 cross entropy loss를 활용합니다.
- 분류 정확도를 확인해서 성능을 측정합니다.

#### 1-3 RNN Time Series prediction

- 주가 예측 데이터셋을 활용한 시계열 데이터 예측을 해결하는 문제입니다.
![image](https://user-images.githubusercontent.com/46701548/139526583-ea8a3881-3285-4c51-9833-afab97ef5a92.png)

- Recurrent Neural Network(RNN)을 활용한 시계열 예측을 해결하는 문제입니다.
- Loss Function으로 MSE loss를 활용합니다.
- 결과 시각화를 통해 성능을 확인합니다.

![image](https://user-images.githubusercontent.com/46701548/139526637-f028efdc-9269-425a-8781-b51a7a0d96ad.png)

## 2. 크롤링 기초

