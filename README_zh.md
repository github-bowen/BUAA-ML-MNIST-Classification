# 使用 SVM 实现 MNIST 数据集图像分类

[TOC]

## 项目介绍

该项目使用 sklearn 工具包中提供的 `sklearn.svm.SVC` 模型进行训练，数据集采用[MNIST数据集](http://yann.lecun.com/exdb/mnist/)。

`train` 目录下的五个训练代码体现了 SVM 超参数调节的过程。（具体见下面**训练过程**）

最终，最优的模型在测试集上的准确率为 **97.79%**。

## 项目目录结构

项目目录如下所示：

```shell
.
├── data
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── models
│   ├── svm_model1.pkl
│   ├── svm_model2.pkl
│   ├── svm_model3.pkl
│   ├── svm_model4.pkl
│   ├── svm_model5.pkl
│   ├── transfer1.pkl
│   ├── transfer2.pkl
│   ├── transfer3.pkl
│   ├── transfer4.pkl
│   └── transfer5.pkl
├── README.md
├── test.py
└── train
    ├── train1.ipynb
    ├── train2.ipynb
    ├── train3.ipynb
    ├── train4.ipynb
    └── train5.ipynb
```

其中：

- `data`目录下存放从作业要求中的[数据地址](http://yann.lecun.com/exdb/mnist/)下载的 MNIST 数据集（经过了解压）。

- `models`目录下存放训练出的模型。该目录下共有 5 组模型，分别对应五个训练代码文件（`train/train{i}.ipynb`，i 为 1-5）。对于第 i 组模型，`svm_model{i}.pkl`为保存的 SVM 模型。

  > 注：由于训练代码中对特征值首先使用 *StandardScaler* 进行了标准化，在测试代码时对于测试数据的特征值也需要进行相同的标准化处理，故需要保存相应的标准化模型`transfer{i}.pkl`。

- `train` 目录下存放 5 组模型的训练代码，文件格式均为 *ipynb*。故运行训练代码需要安装 jupyter 相关依赖，具体安装方法见**环境配置**。

- `test.py` 测试代码，可通过命令行使用，具体使用方法见后文。

## 环境配置

如果不需要运行训练程序（`train/train{i}.ipynb`），运行：

```bash
pip install scikit-learn, numpy, joblib
```

若要运行训练程序，运行：

```bash
pip install scikit-learn, numpy, joblib, jupyter
```

## 代码运行

### 训练模型

打开`train`目录下的 *ipynb* 文件运行所有 cell 即可，生成的模型会保存在 `models` 目录下。

### 测试模型

打开终端或命令，运行如下命令：

- 测试 5 个模型中的一个：

  ```shell
  python test.py $model_number
  ```

  其中：

  `$model_number` 为模型的序号，可取 [1, 2, 3, 4, 5]。

- 测试所有模型：

  ```shell
  python test.py all
  ```

## 测试结果

在 ipython 下运行`python test.py all`来测试所有模型：

```
>ipython
Python 3.8.16 (default, Jan 17 2023, 22:25:28) [MSC v.1916 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.8.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import warnings

In [2]: warnings.filterwarnings('ignore', category=UserWarning)

In [3]: %run test.py all
Start testing model svm_model1:
        Model: SVC(C=10, kernel='poly', max_iter=1000, probability=True)
        Model parameters:  {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
        Predicted values:  [7 2 1 ... 4 5 6]
        True values:  [7 2 1 ... 4 5 6]
        Direct comparison of true values and predicted values:  [ True  True  True ...  True  True  True]
        Accuracy:  0.9717
Model svm_model1 testing completed!

Start testing model svm_model2:
        Model: SVC(C=13, kernel='poly', max_iter=1000, probability=True)
        Model parameters:  {'C': 13}
        Predicted values:  [7 2 1 ... 4 5 6]
        True values:  [7 2 1 ... 4 5 6]
        Direct comparison of true values and predicted values:  [ True  True  True ...  True  True  True]
        Accuracy:  0.9739
Model svm_model2 testing completed!

Start testing model svm_model3:
        Model: SVC(C=19, kernel='poly', max_iter=1000, probability=True)
        Model parameters:  {'C': 19}
        Predicted values:  [7 2 1 ... 4 5 6]
        True values:  [7 2 1 ... 4 5 6]
        Direct comparison of true values and predicted values:  [ True  True  True ...  True  True
True]
        Accuracy:   0.9745
Model svm_model3 testing completed!

Start testing model svm_model4:
        Model: SVC(C=20, kernel='poly', max_iter=5000, probability=True)
        Model parameters: {'C':20}
        Predicted values: [7 2 1 ... 4 5 6]
        True values: [7 2 1 ... 4 5 6]
        Direct comparison of true values and predicted values: [True True True ... True True True]
        Accuracy:   0.9779
Model svm_model4 testing completed!

Start testing model svm_model5:
        Model: SVC(C=30, kernel='poly', max_iter=5000, probability=True)
        Model parameters: {'C':30}
        Predicted values: [7 2 1 ... 4 5 6]
        True values: [7 2 1 ... 4 5 6]
        Direct comparison of true values and predicted values: [True True True ... True True True]
        Accuracy:   0.9779
Model svm_model5 testing completed!
```

可得最佳模型为 `svm_model4` 和 `svm_model5`，在测试集上的准确率为 **97.79%**。

## 训练过程

### 模型1

对于`sklearn.svm.SVC`，一开始选用如下参数进行网格搜索和交叉验证：

```python
# SVM 分类器
svm_model1 = SVC(probability=True, max_iter=1000)

# 网格搜索与交叉验证
param_dict = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svm_model1 = GridSearchCV(svm_model1, param_dict, n_jobs=-1, cv=2)
```

训练后得出交叉验证后在训练集上的最优模型参数（运行测试代码后也会输出）：

```python
模型: SVC(C=10, kernel='poly', max_iter=1000, probability=True)
模型参数:  {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
```

在测试集上准确率为：0.9717

### 模型2

在模型一的基础上进一步调整网格搜索和交叉验证的范围（接下来都只对惩罚系数 C 进行调整）：

```python
# SVM 分类器
svm_model2 = SVC(C=10, kernel='poly', max_iter=1000, gamma='scale', probability=True)

# 网格搜索与交叉验证
param_dict = {
    'C': [7, 8, 9, 10, 11, 12, 13],
     # 'kernel': ['linear', 'rbf', 'poly'],
     # 'gamma': ['scale', 'auto']
}
svm_model2 = GridSearchCV(svm_model2, param_dict, n_jobs=-1, cv=2)
```

训练后得出交叉验证后在训练集上的最优模型参数（运行测试代码后也会输出）：

```python
模型: SVC(C=13, kernel='poly', max_iter=1000, probability=True)
模型参数:  {'C': 13}
```

在测试集上准确率为：0.9739

### 模型3

网格搜索和交叉验证的范围：

```python
# SVM 分类器
svm_model3 = SVC(kernel='poly', max_iter=1000, gamma='scale', probability=True)

# 网格搜索与交叉验证
param_dict = {
    'C': [13, 15, 17, 19, 21],
     # 'kernel': ['linear', 'rbf', 'poly'],
     # 'gamma': ['scale', 'auto']
}
svm_model3 = GridSearchCV(svm_model3, param_dict, n_jobs=-1, cv=2)
```

训练后得出交叉验证后在训练集上的最优模型参数（运行测试代码后也会输出）：

```python
模型: SVC(C=19, kernel='poly', max_iter=1000, probability=True)
模型参数:  {'C': 19}
```

在测试集上准确率为：0.9745

### 模型4

网格搜索和交叉验证的范围：

```python
# SVM 分类器
svm_model4 = SVC(kernel='poly', max_iter=5000, gamma='scale', probability=True)

# 网格搜索与交叉验证
param_dict = {
    'C': [18, 18.5, 19, 19.5, 20],
     # 'kernel': ['linear', 'rbf', 'poly'],
     # 'gamma': ['scale', 'auto']
}
svm_model4 = GridSearchCV(svm_model4, param_dict, n_jobs=-1, cv=2)
```

训练后得出交叉验证后在训练集上的最优模型参数（运行测试代码后也会输出）：

```python
模型: SVC(C=20, kernel='poly', max_iter=5000, probability=True)
模型参数:  {'C': 20}
```

在测试集上准确率为：0.9779

### 模型5

网格搜索和交叉验证的范围：

```python
# SVM 分类器
svm_model5 = SVC(kernel='poly', max_iter=5000, gamma='scale', probability=True)

# 网格搜索与交叉验证
param_dict = {
    'C': [20, 22, 24, 26, 28, 30, 32],
     # 'kernel': ['linear', 'rbf', 'poly'],
     # 'gamma': ['scale', 'auto']
}
svm_model5 = GridSearchCV(svm_model5, param_dict, n_jobs=-1, cv=2)
```

训练后得出交叉验证后在训练集上的最优模型参数（运行测试代码后也会输出）：

```python
模型: SVC(C=30, kernel='poly', max_iter=5000, probability=True)
模型参数:  {'C': 30}
```

在测试集上准确率为：0.9779
