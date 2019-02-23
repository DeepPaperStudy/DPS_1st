## DenseNet

### 1. ResNet vs DenseNet

#### 	ResNet

![](https://i.ibb.co/gtcphZg/1.jpg)

#### 	DenseNet

![](https://i.ibb.co/2c2BDmP/2.jpg)

![](https://i.ibb.co/72FQ3WB/densenet.jpg)

- layer간의 정보 흐름을 최대화하기 위하여 모든 layer를 연결 
  - ResNet은 summation(+)을 사용하였기 때문에 information flow가 옅어지면서 information의 흐름을 방해한다.
  - l개의 레이어는 l개의 인풋을 가지며, l-th layer의 output은 L-l 개의 layer의 인풋으로 사용됨
  - 중요한 것은 ResNet 처럼 summation(+)하지 않고 concatenation한 점
  - 각 layer의 output이 subsequent layer의 input으로 들어가기 때문에 identity function 없이도 정보가 보존.



### 2. Composite function

H_l은 composite function으로 BN, ReLU, Conv의 합성함수

`out = self.conv1(self.relu(self.bn1(x)))`



### 3. Pooling layers(Transition layer)

![](https://camo.githubusercontent.com/ab1cd57673c631f3ee19e875911ea4fb601f4ec6/68747470733a2f2f66696c65732e736c61636b2e636f6d2f66696c65732d7072692f54314a3753434855372d464130514434364a452f64656e7365322e706e673f7075625f7365637265743d33323732313533643635)

down-sampling을 해주기 위해서 Network를 Block 단위로 분리

> down-sampling :  max(or average) pooling 또는 필터 자체의 stride를 높이는 방식으로 이미지의 spatial resolution을 줄이게됩니다. 이렇게 줄여서 한 번에 필터가 볼 수 있는 영역을 좁히면서 해당 이미지의 정보를 압축시키는 것입니다. 출처 : https://jayhey.github.io/deep%20learning/2018/05/26/SqueezeNet/

- feature map의 차원이 바뀌는 경우에는 down-sampling을 해줘야 함
- 중간에 down-sampling을 해주는 layer를 Transition layer라고 함

- Transition layer : `T(x) = 2x2 - avgPooling(BN(1x1-Conv(X)))`



### 4. Growth Rate

![](https://i.ibb.co/qk0ww4t/growh.jpg)

- The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2
- 각 H_l이 k개의 feature map을 만든다면, l-th layer의 input은 k_0 + k * (l-1) 개 의 input을 가짐, 이 때 k_0는 input의 channel, k는 growth rate로 보통 12를 씀
- k가 작아도 잘동작하는 이유는 모든 상태를 input으로 받아오기 때문
- 각 layer는 'collective knowledge'를 합치는 역할을 수행



### 5. Bottleneck layers

- DenseNet-B : `H_l = 3X3-Conv(ReLU(BN(1X1-Conv(ReLU(BN(x))))))`
- output이 k개의 feature map을 만들지만, 그에 비해 input은 굉장히 많으므로 1x1 convolution을 수행하여 feature map을 줄이는 방식으로 연산의 효율성을 증가



### 6. Compression

- transition layer에서도 1x1 convolution을 통해 feature map의 수를 줄여줄 수 있음
- Dense block에서 총 m개의 feature map을 가진다면, 세타 m 개로 줄일 수 있음
- 세타가 1일 경우 feature map의 수가 변하지 않고, feature map < 1 일경우 feature map가 감소되며 이를  DenseNet C라고 부름
- DenseNet-BC : DenseNet B와 C모두 사용



## Implementation Details

![](https://camo.githubusercontent.com/26ad42d777c7cf8dbd25e52cb10666dea4b51274/68747470733a2f2f66696c65732e736c61636b2e636f6d2f66696c65732d7072692f54314a3753434855372d4641303839454755522f666561747572652e706e673f7075625f7365637265743d65383866656638373466)



## Result

![result1](https://camo.githubusercontent.com/25483d7e863a6a2b96afc43ed0407ae8dc387bb1/68747470733a2f2f66696c65732e736c61636b2e636f6d2f66696c65732d7072692f54314a3753434855372d4641304d4332504d5a2f726573756c742e706e673f7075625f7365637265743d32393032393336323334)

![](https://i.ibb.co/k0FdnBc/9841-EFB2-96-DA-40-D7-AF3-A-A087291357-DB-png.jpg)

- parameter가 더 적고, layer의 수가 더 적어도 다른 모델보다 훨씬 성능이 좋다.
- parameter가 적으므로 overfitting의 위험성이 낮다.
- parameter의 수가 증가하더라도 성능이 저하되거나 overfitting이 발생하지는 않는다.



## Reference

- https://github.com/hwkim94/hwkim94.github.io/wiki/Densely-Connected-Convolutional-Networks(2016)
- https://wingnim.tistory.com/39
- https://jayhey.github.io/deep%20learning/2017/10/15/DenseNet_2/
- https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py