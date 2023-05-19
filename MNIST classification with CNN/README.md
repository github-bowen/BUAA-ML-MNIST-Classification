# Implementing MNIST Image Classification using CNN

[TOC]

## Project Introduction

This project builds a convolutional neural network (CNN) using the PyTorch framework to train and test on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

The optimized model achieved an accuracy of **99.55%** on the test set.

## Project Directory Structure

The project directory structure is as follows:

```shell
.
├── cnn.py
├── data
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── dataset.py
├── model.pth
├── README.md
├── README_zh.md
├── test.py
├── train-on-google-colab.ipynb
└── train.py
```

Where:

- The `data` directory contains the MNIST dataset downloaded from the [data link](http://yann.lecun.com/exdb/mnist/) mentioned in the project requirements (after decompression).
- `cnn.py` defines the CNN module.
- `dataset.py` defines the function to load the MNIST dataset from the `/data` directory.
- `model.pth` is the trained model.
- `test.py` contains the testing code.
- `train-on-google-colab.ipynb` is the code run on Google Colab, including all the steps from dataset download to model training and testing. It can be executed directly on Google Colab.
- `train.py` contains the training code.

## Environment Setup

To run the project locally, you need to install common third-party packages such as `torch` and `numpy`.

### Training the Model

Run `train.py` in the project root directory:

```shell
python train.py
```

Open the *ipynb* file in the `train` directory and run all the cells. The generated model will be saved in the `models` directory.

### Testing the Model

Run `test.py` in the project root directory:

```shell
python test.py
```

## Training Process

### Model Definition

The convolutional part of the model uses convolution operations with `stride = 1` and `padding = 0`, and ReLU activation functions. Some convolution layers are followed by max pooling and batch normalization. After the convolution layers, the features are flattened and processed through two fully connected layers. Finally, the softmax function is used for classification. The specific model definition is as follows:

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

i.e.

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

### Hyperparameters and Metric Selection

During the training process, we utilized the cross-entropy loss function as the loss function and employed Adam as the optimizer for backpropagation. To enhance training efficiency, we implemented a strategy of variable learning rates. Specifically, after every 469 * 2 steps, the learning rate was multiplied by 0.1. A total of 13 training epochs were conducted. The detailed configuration information is as follows:

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

### Training Output

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

