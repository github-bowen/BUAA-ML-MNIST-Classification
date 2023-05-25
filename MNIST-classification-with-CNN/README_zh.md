# 使用 CNN 实现 MNIST 数据集图像分类

[TOC]

## 项目介绍

这个项目使用PyTorch框架构建了一个具有5个卷积层和2个线性层的卷积神经网络，对[MNIST数据集](http://yann.lecun.com/exdb/mnist/)进行了训练和测试。

最终，经过优化后的模型在测试集上达到了 **99.55%** 的准确率。

## 项目目录结构

项目目录如下所示：

```shell
.
├── cnn.py
├── data
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── dataset.py
├── model.pth
├── README.md
├── README_zh.md
├── test.py
├── train-on-google-colab.ipynb
├── train.py
```

其中：

- `data`目录下存放从作业要求中的[数据地址](http://yann.lecun.com/exdb/mnist/)下载的 MNIST 数据集（经过了解压）。
- `cnn.py` 为 CNN 模块的定义。
- `dataset.py` 定义了从 `/data` 目录加载 MNIST 数据集的函数。
- `model.pth` 为训练好的模型。
- `test.py` 为测试代码。
- `train-on-google-colab.ipynb` 为在 Google Colab 上运行的代码，代码中包含了数据集的下载、压缩包解压到训练和测试模型**所有流程**，可在 Google Colab 上 **直接运行**。
- `train.py` 为训练代码。

## 环境配置

本地运行需要安装 `torch`、`numpy` 等常见第三方包。

### 训练模型

在项目根目录运行 `train.py`：

```shell
python train.py
```

打开`train`目录下的 *ipynb* 文件运行所有 cell 即可，生成的模型会保存在 `models` 目录下。

### 测试模型

在项目根目录运行 `test.py`：

```shell
python test.py
```

## 训练过程

### 模型定义

模型中的卷积部分采用了`stride = 1`、`padding = 0`的卷积操作，并使用ReLU作为激活函数。部分卷积层后接了最大池化层和批归一化。在卷积层结束后，使用`flatten`将特征展平，然后通过两个全连接层进行处理，最后使用softmax函数进行分类。具体模型定义如下所示：

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(256, 512)
        self.relu6 = nn.ReLU()

        self.linear2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.bn5(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu6(x)

        x = self.linear2(x)
        x = self.softmax(x)
        return x
```

即：

```
CNN(
  (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu1): ReLU()
  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu2): ReLU()
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (relu3): ReLU()
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
  (relu4): ReLU()
  (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bn4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (relu5): ReLU()
  (maxpool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bn5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=256, out_features=512, bias=True)
  (relu6): ReLU()
  (linear2): Linear(in_features=512, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)
```

### 超参数、指标选择

在训练过程中，选择了交叉熵损失函数作为损失函数，并使用Adam作为反向传播的优化器。为了提高训练效率，我们采用了变化学习率的策略。具体而言，每经过469 * 2个步骤，学习率会乘以0.1。总共进行了13轮的训练。下面是详细的配置信息：

```python
# hyper parameters
batch_size = 128
num_epochs = 13
learning_rate0 = 0.001

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

# learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=469 * 2, gamma=0.1)
```

### 训练输出

```
Using device: cuda

Training: 
Epoch [1/13], Step [100/469], Loss: 1.4851, Learning Rate: 0.0010000000
Epoch [1/13], Step [200/469], Loss: 1.4900, Learning Rate: 0.0010000000
Epoch [1/13], Step [300/469], Loss: 1.4912, Learning Rate: 0.0010000000
Epoch [1/13], Step [400/469], Loss: 1.4856, Learning Rate: 0.0010000000
Epoch [2/13], Step [100/469], Loss: 1.4691, Learning Rate: 0.0010000000
Epoch [2/13], Step [200/469], Loss: 1.4921, Learning Rate: 0.0010000000
Epoch [2/13], Step [300/469], Loss: 1.4797, Learning Rate: 0.0010000000
Epoch [2/13], Step [400/469], Loss: 1.4779, Learning Rate: 0.0010000000
Epoch [3/13], Step [100/469], Loss: 1.4694, Learning Rate: 0.0001000000
Epoch [3/13], Step [200/469], Loss: 1.4713, Learning Rate: 0.0001000000
Epoch [3/13], Step [300/469], Loss: 1.4666, Learning Rate: 0.0001000000
Epoch [3/13], Step [400/469], Loss: 1.4848, Learning Rate: 0.0001000000
Epoch [4/13], Step [100/469], Loss: 1.4615, Learning Rate: 0.0001000000
Epoch [4/13], Step [200/469], Loss: 1.4764, Learning Rate: 0.0001000000
Epoch [4/13], Step [300/469], Loss: 1.4612, Learning Rate: 0.0001000000
Epoch [4/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0001000000
Epoch [5/13], Step [100/469], Loss: 1.4612, Learning Rate: 0.0000100000
Epoch [5/13], Step [200/469], Loss: 1.4679, Learning Rate: 0.0000100000
Epoch [5/13], Step [300/469], Loss: 1.4612, Learning Rate: 0.0000100000
Epoch [5/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0000100000
Epoch [6/13], Step [100/469], Loss: 1.4688, Learning Rate: 0.0000100000
Epoch [6/13], Step [200/469], Loss: 1.4690, Learning Rate: 0.0000100000
Epoch [6/13], Step [300/469], Loss: 1.4686, Learning Rate: 0.0000100000
Epoch [6/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0000100000
Epoch [7/13], Step [100/469], Loss: 1.4765, Learning Rate: 0.0000010000
Epoch [7/13], Step [200/469], Loss: 1.4693, Learning Rate: 0.0000010000
Epoch [7/13], Step [300/469], Loss: 1.4637, Learning Rate: 0.0000010000
Epoch [7/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0000010000
Epoch [8/13], Step [100/469], Loss: 1.4620, Learning Rate: 0.0000010000
Epoch [8/13], Step [200/469], Loss: 1.4618, Learning Rate: 0.0000010000
Epoch [8/13], Step [300/469], Loss: 1.4612, Learning Rate: 0.0000010000
Epoch [8/13], Step [400/469], Loss: 1.4613, Learning Rate: 0.0000010000
Epoch [9/13], Step [100/469], Loss: 1.4612, Learning Rate: 0.0000001000
Epoch [9/13], Step [200/469], Loss: 1.4612, Learning Rate: 0.0000001000
Epoch [9/13], Step [300/469], Loss: 1.4615, Learning Rate: 0.0000001000
Epoch [9/13], Step [400/469], Loss: 1.4690, Learning Rate: 0.0000001000
Epoch [10/13], Step [100/469], Loss: 1.4686, Learning Rate: 0.0000001000
Epoch [10/13], Step [200/469], Loss: 1.4693, Learning Rate: 0.0000001000
Epoch [10/13], Step [300/469], Loss: 1.4668, Learning Rate: 0.0000001000
Epoch [10/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0000001000
Epoch [11/13], Step [100/469], Loss: 1.4689, Learning Rate: 0.0000000100
Epoch [11/13], Step [200/469], Loss: 1.4633, Learning Rate: 0.0000000100
Epoch [11/13], Step [300/469], Loss: 1.4612, Learning Rate: 0.0000000100
Epoch [11/13], Step [400/469], Loss: 1.4690, Learning Rate: 0.0000000100
Epoch [12/13], Step [100/469], Loss: 1.4640, Learning Rate: 0.0000000100
Epoch [12/13], Step [200/469], Loss: 1.4612, Learning Rate: 0.0000000100
Epoch [12/13], Step [300/469], Loss: 1.4692, Learning Rate: 0.0000000100
Epoch [12/13], Step [400/469], Loss: 1.4616, Learning Rate: 0.0000000100
Epoch [13/13], Step [100/469], Loss: 1.4676, Learning Rate: 0.0000000010
Epoch [13/13], Step [200/469], Loss: 1.4689, Learning Rate: 0.0000000010
Epoch [13/13], Step [300/469], Loss: 1.4619, Learning Rate: 0.0000000010
Epoch [13/13], Step [400/469], Loss: 1.4612, Learning Rate: 0.0000000010

Accuracy on test set: 99.55 %
```