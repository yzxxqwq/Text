***

## Conda

- `conda create`：创建一个新的conda环境。

- `conda activate`：激活conda环境。

- `conda deactivate`：停用conda环境。

- `conda install`：安装软件包。

- `conda update`：更新软件包。

- `conda list`：列出当前环境中安装的软件包。

- `conda remove`：删除软件包。

- `conda info`：查看conda的版本信息。

- `conda search`：搜索软件包。

- 查看GPU的数量。`nvidia-smi`t

- `orch.cuda.memory_summary()` #查看GPU内存使用情况 

- `torch.cuda.memory_allocated()` # 查看已分配的GPU内存

-  `torch.cuda.memory_cached()` # 查看已缓存的GPU内存

- 在Anaconda中安装PyTorch：

  1. 首先，确保你已经在Anaconda中创建了一个虚拟环境（如果你没有创建，可以使用conda create命令创建）。
  2. 在命令行中输入：

  `conda install pytorch torchvision -c pytorch`

  这将安装最新版本的PyTorch和torchvision包。

  1. 如果你想安装特定版本的PyTorch，可以使用以下命令：

  `conda install pytorch=1.7.0 torchvision=0.8.1 -c pytorch`

  这将安装PyTorch 1.7.0和torchvision 0.8.1。

  1. 如果你想在CUDA支持的GPU上运行PyTorch，可以使用以下命令：

  `conda install pytorch torchvision cudatoolkit=11.0 -c pytorch`

  这将安装最新版本的PyTorch，torchvision和CUDA 11.0工具包。

  注意：这些命令假定你已经在系统上安装了CUDA。如果你没有安装CUDA，可以使用以下命令安装CPU版本的PyTorch：

  `conda install pytorch torchvision cpuonly -c pytorch`
  
  计算机上安装了CUDA 12.0，则可以使用PyTorch 1.8及更高版本。

***

**修改库版本**

1. 创建新的虚拟环境：在Anaconda中，你可以使用conda命令来创建新的虚拟环境。例如，如果你希望创建名为“pytorch-env”的虚拟环境，则可以使用以下命令：

```
Copy code
conda create --name pytorch-env
```

然后，你可以使用以下命令来激活新创建的虚拟环境：

```
Copy code
conda activate pytorch-env
```

1. 在新的虚拟环境中安装PyTorch：在新的虚拟环境中，你可以使用以下命令来安装所需版本的PyTorch：

```
Copy code
conda install pytorch==X.Y
```

其中X.Y表示你希望安装的PyTorch版本。例如，如果你希望安装PyTorch 1.10，则可以使用以下命令：

```
Copy code
conda install pytorch==1.10
```

1. 退出虚拟环境：如果你希望退出虚拟环境，可以使用以下命令：

```
Copy code
conda deactivate
```

总的来说，如果你希望在Anaconda中更改PyTorch的版本信息，可以使用conda命令创建新的虚拟环境，然后在新的虚拟环境中使用conda命令安装所需版本的PyTorch。最后，你可以使用conda命令退出虚拟环境

***

你可以在Anaconda的虚拟环境内更改PyTorch的版本信息。可以使用以下步骤之一来实现：

1. 激活需要修改的虚拟环境：如果你希望修改某个虚拟环境内的PyTorch版本，那么首先要激活该虚拟环境。例如，如果你希望修改名为“pytorch-env”的虚拟环境内的PyTorch版本，则可以使用以下命令：

```
Copy code
conda activate pytorch-env
```

1. 卸载旧版本的PyTorch：在虚拟环境内，你可以使用以下命令来卸载旧版本的PyTorch：

```
Copy code
conda remove pytorch
```

1. 安装新版本的PyTorch：在虚拟环境内，你可以使用以下命令来安装新版本的PyTorch：

```
Copy code
conda install pytorch==X.Y
```

其中X.Y表示你希望安装的PyTorch版本。例如，如果你希望安装PyTorch 1.10，则可以使用以下命令：

```
Copy code
conda install pytorch==1
```

## 基础知识

#### 优化算法

当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。本节使用的线性回归和平方误差刚好属于这个范畴。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）

***

在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。它的算法很简单：先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）BB，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。

在训练本节讨论的线性回归模型的过程中，模型的每个参数将作如下迭代：

***

#### 数据样本的初始化

- 

***

5.样本的初始化处理

```

```



***



2.下面是线性回归的矢量计算表达式的实现。我们使用`dot`函数做矩阵乘法

```
def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    return nd.dot(X, w) + b
```

***

3.定义损失函数

1）平方损失函数

```
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

***

#### MEXNET自动求梯度

MEXNET提供的`autograd`模块来自动求梯度

1.首先需要导入`autograd`包
2.创建一个输入函数的`NDArray`
3.调用`attach_grad`函数来保存函数f(x)相对于输入值x的梯度
4.在`autograd.record()`作用域中定义函数f(x)，便于存储f(x)，从而计算梯度
5.调用backward()函数来执行反向传播，从而计算函数f(x)相对于输入x的导数

```
from mxnet import autograd,nd
#获得初始X自变量数据集
x = nd.arange(4).reshape((4,1))
#调用attach_grad函数为x的梯度申请内存
x.attach_grad()
#为了减少计算和内存开销，默认条件下MXNet不会记录用于求梯度的计算。我们需要调用record函数来要求MXNet记录与求梯度有关的计算
#定义前向传播的函数，保存函数计算梯度
with autograd.record()：
	y = 2 * nd.dot(x.T,x)
#调用反向传播
y.backward()
```

***



![](C:\Users\11984\Desktop\image-20221227170903933.png)



***

#### softmax运算

```
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制
```

在下面的函数中，矩阵`X`的行数是样本数，列数是输出个数。为了表达样本预测各个输出的概率，softmax运算会先通过`exp`函数对每个元素做指数运算，再对`exp`矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。

***

#### 计算分类准确率

给定一个类别的预测概率分布`y_hat`，我们把预测概率最大的类别作为输出类别。如果它与真实类别`y`一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。

为了演示准确率的计算，下面定义准确率`accuracy`函数。其中`y_hat.argmax(axis=1)`返回矩阵`y_hat`每行中最大元素的索引，且返回结果与变量`y`形状相同。我们在[“数据操作”](http://zh.gluon.ai/chapter_prerequisite/ndarray.html)一节介绍过，相等条件判别式`(y_hat.argmax(axis=1) == y)`是一个值为0（相等为假）或1（相等为真）的`NDArray`。由于标签类型为整数，我们先将变量`y`变换为浮点数再进行相等条件判断。

***

#### 权重衰减/正则化

可见，L2范数正则化令权重w1和w2先自乘小于1的数，再减去不含惩罚项的梯度。因此，L2范数正则化又叫权重衰减。权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，这可能对过拟合有效。实际场景中，我们有时也在惩罚项中添加偏差元素的平方和。

***

#### Terms

GitHub上的README文件是一个文本文件，用于向其他用户介绍项目的内容和如何使用项目。它通常位于项目的根目录中，并以"README"命名。

README文件通常包含以下内容：

- 项目的目的：简要介绍项目的目的和功能
- 如何安装和使用项目：提供有关如何安装和使用项目的说明
- 项目的依赖关系：列出项目所依赖的其他库、工具或软件
- 如何贡献：提供有关如何贡献代码或建议的信息
- 许可证信息：指定可以使用项目的许可证

GitHub上的README文件通常使用Markdown语法书写，使其易于阅读和维护。通过在项目主页上显示README文件，可以向其他用户提供有关项目的信息，以便他们了解项目并决定是否使用它

CUDA（Compute Unified Device Architecture）是 NVIDIA 的计算平台和编程模型，用于让计算机上的图形处理器（GPU）执行大量并行计算任务。这使得可以使用 GPU 加速计算应用程序的运行时间，特别是对于那些需要执行大量并行运算的应用程序，如机器学习和深度学习应用程序。

Miniconda是一个轻量级的Python发行版，包含了conda、Python和一些必要的库。它不包含任何与数据科学、机器学习和科学计算相关的库。

Anaconda是一个完整的Python发行版，包含了conda、Python和超过250个科学库和工具包。它非常适用于数据科学、机器学习和科学计算领域。

通常，如果你只需要使用conda管理你的Python环境，你可以使用Miniconda。如果你需要使用许多科学库和工具包，你可以使用Anaconda。

解释器是一种程序，用于执行Python代码。它负责读取代码，解释其含义，并执行相应的操作。环境指的是解释器的运行环境，包括了安装的第三方包和已经导入的模块。不同的环境可以有不同的解释器和安装的包，因此在使用Python时，通常需要确定使用的解释器和环境。在PyCharm中，可以使用解释器管理器来管理不同的解释器和环境，并在项目中切换使用不同的解释器和环境。

MLP（多层感知机）是一种深度学习模型，其包含若干个全连接层（也称为感知层）。在每个全连接层中，有若干个神经元，每个神经元都与上一层的所有神经元有连接。因此，MLP可以通过线性变换和非线性激活函数来学习复杂的数据表示。

CNN（卷积神经网络）也是一种深度学习模型，其包含若干个卷积层和池化层。卷积层通过卷积核（也称为滤波器）来对输入图像进行滤波，从而提取图像的特征。池化层通过池化窗口将图像的多个位置的信息合并起来，从而减少数据的维度和参数的数量。

DCNN是指卷积神经网络（Convolutional Neural Network, CNN）。CNN是一种用于图像处理和分类的神经网络模型，它通过使用卷积操作来提取图像中的特征，并使用这些特征来进行分类。CNN通常用于图像分类、识别和分割等任务，也可以用于视频分析、自然语言处理等领域。

# 基本代码公式

1.`mx.nd.random.normal(loc,scale,shape)`

生成样本根据由loc*（均值）和*scale（标准差）参数化的正态分布分布随机数。

```
from mxnet import nd
from time import time
#我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
# 权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```

2.`numpy.asscalar( )`

将大小为1的数组转换为其等效的标量(scalar)。

***

3.独热编码`pd.get_dummies`

`pd.get_dummies` 函数是 `pandas` 库中的一个重要函数，用于将数据帧中的非数值型特征进行 one-hot 编码。

```
pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

```

- `data`：需要进行 one-hot 编码的数据帧。
- `prefix`：新特征的前缀。可以是任意的字符串。
- `prefix_sep`：前缀和原本特征名之间的分隔符。可以是任意的字符串。
- `dummy_na`：布尔值，用于指定是否将缺失值也当作合法的特征值，并为其创建一个指示特征。如果设置为 True，则会为缺失值创建一个指示特征；如果设置为 False，则不会为缺失值创建指示特征。
- `columns`：需要进行 one-hot 编码的特征名的列表。如果不指定，则会对数据帧中的所有非数值型特征进行 one-hot 编码。
- `sparse`：布尔值，用于指定是否使用稀疏矩阵存储 one-hot 编码后的数据。
- `drop_first`：布尔值，用于指定是否删除一个非数值型特征的第一个指示特征。如果设置为 True，则会删除一个非数值型特征的

***

4.缺失值替换`df.fillna()`

```
全零初始化
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

`fillna` 是 `pandas` 库中的一个重要函数，用于将数据帧中的缺失值替换为指定的值。

使用方法如下：

```
df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
```

- `df`：需要替换缺失值的数据帧。
- `value`：替换缺失值的值。可以是任意的 Python 对象，例如数字、字符串、布尔值、列表等。
- `method`：替换缺失值的方法。可以设置为 'ffill' 或 'bfill'，表示使用前向填充或后向填充的方法替换缺失值。
- `axis`：替换缺失值的方向。可以设置为 0 或 1，表示沿着行或列的方向替换缺失值。
- `inplace`：布尔值，用于指定是否在原数据帧上直接修改。如果设置为 True，则会在原数据帧上修改；如果设置为 False，则会返回一个修改后的数据帧。默认值为 False。
- `limit`：整数值，用于指定前向或后向填充时，填充的最大数量。
- `downcast`：降维参数，用于指定填充后的数据是否需要降维。
- `**kwargs`：可选参数，传入需要传入函数的关键字参数。

5.`x.view(shape)`函数是PyTorch中的一个常用函数，用于将一个张量的形状调整为新的形状。

其中，x表示要调整形状的张量，shape是一个元组，表示新的形状。当shape中的某一维设置为-1时，表示自动计算该维的大小，使得张量的总元素数不变。

使用view函数可以方便地将张量调整为指定的形状，并且不会改变张量的内存地址，也不会复制张量的数据。例如，可以使用view函数将一个二维张量调整为一维张量，或者将一个三维张量调整为二维张量等。

```
import torch

# 创建一个3*3的二维张量
x = torch.arange(9,dtype=torch.float32).reshape((3,3))
print(x)  # 输出：tensor([[0., 1., 2.],
           #        [3., 4., 5.],
           #        [6., 7., 8.]])

# 使用view函数将x调整为一维张量
y = x.view(-1)
print(y)  #
```

***

6.`torchvision.transforms.ToTensor`

torchvision.transforms.ToTensor函数是pytorch中的一种数据预处理方式，用于将图像数据转换为张量（Tensor）。

这个函数会将每张图像的像素值从[0, 255]范围内的整数转换为[0.0, 1.0]范围内的浮点数，并将每个像素值单独放在一个位置上。例如，一张灰度图像可以被转换为一个2维张量，而一张彩色图像则可以被转换为一个3维张量，分别对应着红、绿、蓝三种颜色通道。

这个函数在深度学习中常被用于加载图像数据集，例如MNIST数据集，因为张量比较容易被神经网络处理，而图像数据本身则是像素矩阵的形式。

# pytorch

## 多层感知机 （multilayer perceptron，MLP）

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。图3.3展示了一个多层感知机的神经网络图。![带有隐藏层的多层感知机。它含有一个隐藏层，该层中有5个隐藏单元](http://zh.gluon.ai/_images/mlp.svg)





***

### 实例解析

#### 1.层与块

module ：n.模块/单元

```
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)
```

##### nn.Module

在 PyTorch 中，你可以通过创建 nn.Module 子类来定义自己的神经网络。nn.Module 是 PyTorch 中定义神经网络的基类，它包含了神经网络中常用的方法和属性。你可以通过继承 nn.Module 类，并实现它的 forward 函数来定义自己的神经网络。

例如：你可以这样子来实现自己的神经网络：

```
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 定义第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 定义第二层全连接层
        
    def forward(self, x):
```

#### 线性层nn.Linear

nn.Linear 层是一个线性层，它实现了线性变换 y = wx + b，其中 w 是一个维度为 (out_features, in_features) 的权重矩阵，b$是一个维度为 (out_features,) 的偏差向量。x是输入的一个维度为 (batch_size, in_features) 的张量，y$是输出的一个维度为 (batch_size, out_features) 的张量。你可以通过调用 forward 函数来执行前向计算，获得输出张量。

##### 如何查看nn.Linner的系数w和b

```
linear = nn.Linear(in_features=20, out_features=256)
w, b = linear.weight, linear.bias
```

##### 如何修改nn.Linear的系数w和b

```
linear.weight = nn.Parameter(torch.randn(256, 20))
linear.bias = nn.Parameter(torch.randn(256))
```

在 PyTorch 中，nn.Parameter 是一个特殊的张量，它与模型的参数有关。当你使用一个 nn.Module 的子类创建模型时，它的参数需要被包装成 nn.Parameter 类型的对象，这样 PyTorch 才能自动将这些参数添加到模型的参数列表中，并将它们进行梯度计算。

nn.Module 类提供了一些方法，如 named_parameters 和 parameters，你可以使用这些方法来查看模型的参数。这些方法会返回模型的参数的一个迭代器，每个参数都是一个 (名字, 参数) 的元组，其中名字是参数的属性名，参数是 nn.Parameter 类型的对象。

#### 2.自定义块

自定义块和顺序块是 PyTorch 中的两种模块。自定义块是指用户自己定义的模块，它可以是一个线性变换层，也可以是一个复杂的神经网络模型。顺序块是指包含多个自定义块的序列，并按照一定顺序依次调用这些自定义块的模块。

具体来说，自定义块是一种可以在模型中自由使用的模块，可以用于实现各种各样的计算。例如，你可以使用自定义块来实现一个线性变换层，也可以使用自定义块来实现一个卷积神经网络模型。

顺序块是一种将多个自定义块组合在一起的模块。它可以按照一定顺序依次调用这些自定义块，并将输出数据传递给下一个自定义块。这样，就可以用顺序块来构建复杂的神经网络模型。

```
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的__init__函数
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
        
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
# 实例化多层感知机的层，然后在每次调用正向传播函数调用这些层
net = MLP()
X = torch.rand(2,20)
net(X)
```

这段代码中定义了一个多层感知机（MLP）的网络结构。

在这个网络结构中，有两个全连接层：一个隐藏层和一个输出层。隐藏层的输入维度为 20，输出维度为 256；输出层的输入维度为 256，输出维度为 10。

1. 在这段代码中，定义的类名为 `MLP`，它继承了 PyTorch 中的 `nn.Module` 类。这种类的定义方式在 Python 中是很常见的，它允许我们创建新的类，并从现有的类中**继承属性和方法**。

2. 在这段代码中，`super().__init__()` 用来调用父类的 `__init__` 函数。

   在 Python 中，类的构造函数通常命名为 `__init__`，它在类的实例创建时被调用。在这里，调用父类的 `__init__` 函数的目的是初始化父类的内容，例如父类的属性。

   在 Python 中，如果一个子类想要调用父类的方法，可以使用 `super()` 函数。例如，在这段代码中，子类 `MLP` 继承了 `nn.Module` 类，并在定义自己的 `__init__` 函数时调用了父类的 `__init__` 函数。

3. `nn.Linear(20, 256)` 这个函数用来创建一个全连接层，它的输入维度为 20，输出维度为 256。

这个函数的输出结果是一个全连接层的实例，它可以被用来计算输入数据的线性变换。例如：在 PyTorch 中，可以使用全连接层的实例的 `forward` 方法来计算线性变换的输出

```
import torch

# 创建全连接层
fc = nn.Linear(20, 256)

# 创建输入数据
x = torch.randn(128, 20)

# 计算线性变换的输出
y = fc.forward(x)
print(y.shape)  # 输出: torch.Size([128, 256])


```

```
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x
```



***

#### 3.顺序块

```
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block # block 本身作为它的key，存在_modules里面的为层，以字典的形式
            
    def forward(self, X):
        for block in self._modules.values():
            print(block)
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)
```

1. 字典的使用方法：创建了一个空字典，然后使用 `d[key] = value` 的方式将几个键值对添加到字典中。在这里，`self._modules` 是一个字典，用于保存自定义块（即 PyTorch 中的模块）。使用 `self._modules[block] = block` 这句话，就是将自定义块本身作为字典的 key，并将自定义块本身作为字典的 value。

   

2. 这段代码中定义了一个名为 `MySequential` 的类，它继承了 PyTorch 中的 `nn.Module` 类。

   这个类的构造函数 `__init__` 中，使用了可变长度参数 `*args`，这意味着在调用这个函数时，可以传入任意多个参数。

   这里利用一个for循环读取输入的可变参数(激活函数)，放入一个类为self._modules的字典里面

   

3. 在 Python 中，`*args` 是一种常见的参数定义方法，表示接收一个可变数量的参数。

   具体来说，`*args` 表示一个可变数量的位置参数，它会将调用函数时传递的所有位置参数都组装成一个元组传递给函数。

   

4. 在这个类中，还定义了一个名为 `forward` 的方法，这个方法用来计算神经网络的前向传播输出。这个方法中，遍历了  `self._modules` 字典中的所有层，并使用这些层依次计算输入数据的线性变换。

   具体来说，对于每一个层 `block`，它都会接收输入数据 `X`，并计算输出数据 `Y`。然后将输出数据 `Y` 作为下一个层的输入数据，继续计算。最后，这个方法返回输入数据经过所有层计算后的结果。

   这个类的作用是什么呢？它允许我们以一种简单的方式创建一个包含多个层的神经网络模型。

#### 4.正向传播块

```
# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight + 1))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
X = torch.rand(2,20)
net(X)
```

### 5.参数管理

nn.Module 类中有一个名为 _parameters 的成员变量，它是一个字典，用于存储模型中所有参数。这个字典中的每一项都是一个 nn.Parameter 类型的对象，包含了参数的名称和值。

```
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
print(net(X))
print(net[2].state_dict()) # 访问参数，net[2]就是最后一个输出层
print(type(net[2].bias)) # 目标参数
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None) # 还没进行反向计算，所以grad为None
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 一次性访问所有参数         
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 0是第一层名字，1是ReLU，它没有参数
print(net.state_dict()['2.bias'].data) # 通过名字获取参数
```

在模块中Parameter

```
`print(net[2].bias)`
输出`tensor([0.3129], requires_grad=True)`

`print(net[2].bias.data)`
输出tensor([0.3129])
```


``







## 如何保留训练模型

如果想保存训练好的模型，可以使用 PyTorch 的 `torch.save` 函数将模型的参数保存到文件中。

```
torch.save(model.state_dict(), 'model.pt')
```

保存的模型文件是一个二进制文件，包含了模型的结构和参数。

如果想加载这个模型，可以使用 PyTorch 的 `torch.load` 函数将模型的参数加载到内存中。

```
model = MLP(20, 256, 10) model.load_state_dict(torch.load('model.pt'))
```

除了使用 `torch.save` 和 `torch.load` 函数之外，还可以使用 PyTorch 提供的其他保存和加载函数来保存和加载模型。例如：

- 使用 `torch.save` 函数保存模型时，还可以使用参数 `pickle_module` 来指定使用哪个模块来进行序列化，默认情况下使用 Python 的 pickle 模块。你可以使用 `torch.save(model, 'model.pt', pickle_module=dill)` 的方式来使用 dill 模块进行序列化。
- 使用 `torch.save` 函数保存模型时，还可以使用参数 `zip_file` 来将模型文件压缩成 zip 文件，使用方法与普通文件一样。例如，使用 `torch.save(model, 'model.zip', zip_file=True)` 可以将模型文件压缩成 `model.zip` 文件。
- 使用 `torch.load` 函数加载模型时，还可以使用参数 `map_location` 来指定加载模型时使用的设备，例如 CPU 或 GPU

***

# CUBA

若要在 Python 中调用 GPU，需要安装适用于 Python 的 GPU 驱动程序，并使用支持 GPU 的 deep learning 框架，如 PyTorch 或 TensorFlow。

首先，需要确保电脑具有 NVIDIA GPU 并已安装 NVIDIA GPU 驱动程序。然后，可以使用 conda 命令来安装 PyTorch，并在 Python 中使用 PyTorch 来访问 GPU。例如，可以在命令行中输入以下命令来安装 PyTorch：

`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`

然后，可以在 Python 代码中使用以下语句将数据移动到 GPU：

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu") x = x.to(device)`

还可以使用 PyTorch 的 nn.Module 类来定义的模型，并使用 .to() 方法将模型移动到 GPU。例如：

`model = MyModel().to(device)`

此外，还可以使用 TensorFlow 作为您的 deep learning 框架，并使用相似的方法来访问 GPU。

在使用 GPU 时，请注意训练模型所需的内存大小。若 GPU 内存不足，则可能无法将数据或模型加载到 GPU 中。在这种情况下，可能需要调整您的 batch_size 或使用更小的模型。

在Python中，可以使用以下代码来检查你的系统是否有可用的GPU：

**pytorch**

```
import torch

# 设置默认的device为GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型转换为GPU张量
model = model.to(device)

# 将数据转换为GPU张量
data = data.to(device)

# 运行模型
output = model(data)
```

***

在使用PyTorch和CUDA进行深度学习时，有一些注意事项需要遵守。其中一个重要的注意事项是，PyTorch和CUDA有版本要求。

在安装PyTorch时，你需要确保你安装的版本与你的CUDA版本相匹配。例如，如果你的CUDA版本为10.1，则你应该安装PyTorch版本1.7或更高版本。如果你的CUDA版本为11.0，则你应该安装PyTorch版本1.8或更高版本。你可以在PyTorch官网上查看最新的版本要求信息。

同时，你还需要确保你的CUDA版本与你的显卡相匹配。例如，如果你的显卡支持CUDA版本10.2，则你应该使用CUDA版本10.2或更高版本。如果你的显卡只支持CUDA版本9.0，则你应该使用CUDA版本9.0或更低版本。

总的来说，如果你希望使用PyTorch和CUDA进行深度学习，则必须确保你的PyTorch版本与你的CUDA版本相匹配，并且你的CUDA版本与你的显卡相匹配。

# 卷积神经网络

**卷积核Convolution kernel**

![image-20221229163016302](C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20221229163016302.png)

![](C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20221229211022883.png)

***

![image-20221229223137632](C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20221229223137632.png)

***



## 卷积神经网络

在卷积神经网络中，卷积层和池化层通常被组合使用来提取图像的特征。通常，卷积层会使用多个卷积核来提取图像的不同特征，并通过卷积操作来保留这些特征之间的空间关系。池化层则会使用池化操作来对这些特征进行采样，缩小图像的尺寸，同时保留关键的特征。

卷积层和池化层的主要作用是提取图像的特征，因此通常会将多个卷积层和池化层堆叠在一起，形成卷积层模块。在卷积层模块之后，通常会使用全连接层（也叫多层感知机）进行分类。全连接层通常会使用较多的神经元，将卷积层模块提取的特征进行高度抽象的组合，从而对图像进行分类。最后，通常会使用softmax函数将网络的输出转换成概率分布，表示图像属于每个类别的概率。

因此，在做图像识别时，通常会先使用卷积层和池化层来提取图像的特征，然后使用全连接层进行分类，最后使用softmax函数输出分类的

### 卷积层

#### **从全连接到卷积层**

卷积神经网络（Convolutional Neural Network，CNN）与普通神经网络之间的一个主要区别在于它们使用的层类型不同。

普通神经网络通常使用全连接层，在全连接层中，所有输入神经元与所有输出神经元之间都有连接。也就是说，输入和输出之间的所有可能连接都会被建立。这使得全连接层适合处理向量数据，但不适合处理具有空间结构的数据，例如图像。

卷积神经网络使用卷积层和池化层来处理具有空间结构的数据。卷积层通过使用卷积核（通常称为滤波器）来对输入数据进行空间卷积，从而提取输入数据的有用信息。池化层则使用池化窗口对输入数据进行下采样，从而缩小输入数据的尺寸。

**卷积层和全连接的区别**

二维卷积层经常用于处理图像，与此前的全连接层相比，它主要有两个优势：

**一是全连接层把图像展平成一个向量**，在输入图像上相邻的元素可能因为展平操作不再相邻，网络难以捕捉局部信息。而卷积层的设计，天然地具有提取局部信息的能力。

**二是卷积层的参数量更少**。不考虑偏置的情况下，一个形状为![(c_i, c_o, h, w)](https://math.jianshu.com/math?formula=(c_i%2C%20c_o%2C%20h%2C%20w))的卷积核的参数量是![c_i \times c_o \times h \times w](https://math.jianshu.com/math?formula=c_i%20%5Ctimes%20c_o%20%5Ctimes%20h%20%5Ctimes%20w)，与输入图像的宽高无关。假如一个卷积层的输入和输出形状分别是![(c_1, h_1, w_1)](https://math.jianshu.com/math?formula=(c_1%2C%20h_1%2C%20w_1))和![(c_2, h_2, w_2)](https://math.jianshu.com/math?formula=(c_2%2C%20h_2%2C%20w_2))，如果要用全连接层进行连接，参数数量就是![c_1 \times c_2 \times h_1 \times w_1 \times h_2 \times w_2](https://math.jianshu.com/math?formula=c_1%20%5Ctimes%20c_2%20%5Ctimes%20h_1%20%5Ctimes%20w_1%20%5Ctimes%20h_2%20%5Ctimes%20w_2)。使用卷积层可以以较少的参数数量来处理更大的图像

***

#### **卷积层-四维张量**

批量维（batch dimension）和通道数（channel dimension）是深度学习中常见的两种维度。

第一维表示批量维，第二维表示通道数，第三维表示高度，第四维表示宽度。

批量维是数据样本维，也就是一次输入的数据样本数。对于每个输入样本，模型会输出一个输出结果。

通道数是指数据的通道数量。对于图像数据，通道数通常指图像的颜色通道数，例如红、绿、蓝三个颜色通道。对于灰度图像，通道数通常为 1，因为只有一个灰度值。

例如，对于一张黑白图像，其四维张量的形状为（1, 1, 3, 3），其中第一维表示批量维，第二维表示通道数，第三维表示高度，第四维表示宽度。这张图像有 1 个样本，1 个通道，高度为 3 像素，宽度为 3 像素。

对于一张彩色图像，其四维张量的形状为（1, 3, 3, 3），其中第一维表示批量维，第二维表示通道数，第三维表示高度，第四维表示宽度。这张图像有 1 个样本，3 个通道（红、绿、蓝），高度为 3 像素。

***

##### 通道数

在卷积神经网络中，一般使用一个卷积核来处理一个通道。因此，对于一张有 3 个通道的图像，通常会使用 3 个卷积核来处理这三个通道。

卷积核的数量通常是固定的，例如对于一个灰度图像，可以使用一个卷积核；对于一张彩色图像，可以使用 3 个卷积核（分别用于处理红、绿、蓝三个通道）。

然而，在某些情况下，可能会使用多于 3 个卷积核来处理彩色图像的通道。例如，可以使用更多的卷积核来提取图像的更多信息。

但是，需要注意的是，使用更多的卷积核会增加模型的复杂度，并可能导致过拟合。因此，在设计卷积神经网络时，需要考虑如何平衡模型的复杂度和泛化能力。

##### 卷积中的点积

卷积操作中的点积和矩阵点积（通常称为矩阵乘法或矩阵的点积）是不同的操作。

卷积操作中的点积是指将两个矩阵的对应元素相乘，然后将所有乘积的和相加。在这种情况下，两个矩阵的大小应该相同。

矩阵点积是指将两个矩阵相乘，得到一个新矩阵。矩阵点积的计算方法是，对于输出矩阵的每个位置，将输入矩阵的对应行的所有元素与输入矩阵的对应列的所有元素相乘，然后将所有乘积的和相加。要计算矩阵点积，输入矩阵的列数必须等于另一个矩阵的行数。

***

#### 卷积核

1. 在卷积神经网络中，卷积核的作用是在卷积层中进行卷积操作时进行特征提取。在卷积层中，卷积核会与输入图像的小区域（通常是图像的矩形小块）进行卷积运算，从而提取出图像的不同特征。例如，使用边缘检测的卷积核可以提取出图像的边缘信息，使用高斯滤波器的卷积核可以使图像模糊，使用Sobel算子的卷积核可以提取出图像的梯度信息等。

   因此，在卷积神经网络中，卷积核是由用户自定义的，其作用是提取图像的特定特征。通常，卷积核的设计会根据需要提取的图像特征而不同。在卷积神经网络的训练过程中，卷积核的参数也会被学习，从而更好地提取图像的特征。

2. 在卷积神经网络的训练过程中，卷积核的参数也会通过梯度下降法进行调整。梯度下降法是一种用来优化模型参数的常用方法，它通过不断调整参数来最小化模型的损失函数。

   在卷积神经网络的训练过程中，卷积核的参数也会被调整，使模型能够更好地提取出图像的特征。这个过程是通过求损失函数对卷积核参数的偏导数来实现的。求出偏导数后，就可以使用梯度下降法来调整卷积核的参数，使得损失函数的值越来越小。

3. 简单来说，卷积核参数的梯度下降的原理就是利用损失函数的偏导数来调整参数，使得损失函数的值越来越小。

   卷积神经网络的训练过程与其他神经网络的训练过程类似，也是通过反向传播算法和梯度下降法来调整模型的参数，使模型的预测结果更加准确。在卷积神经网络的训练过程中，卷积核的参数也会被调整，从而使模型能够更好地提取出图像的特征。

举个例子，假设你正在训练一个卷积神经网络，用来识别手写数字。在训练的过程中，你的模型可能会发现，用来检测数字的边缘的卷积核效果很差，而用来检测数字的轮廓的卷积核效果很好。在这种情况下，你的模型就会调整用来检测轮廓的卷积核的参数，使其能够更好地提取出数字的轮廓信息。



#### 填充

在卷积神经网络中，卷积层的填充是指在输入张量的边缘填充额外的值，以便能够使用卷积核在输入的四周进行卷积运算。

卷积层的填充有两种常见的方法：

- 填充 0（zero padding）：在输入张量的边缘填充 0。这种方法可以使得卷积后的输出维度与输入维度相同。
- 填充非零值（non-zero padding）：在输入张量的边缘填充非零值。这种方法可以使得卷积后的输出维度与输入维度不同。

卷积层的填充通常用于调整网络的结构，以满足特定的应用需求。例如，使用填充 0 可以使得卷积后的输出维度与输入维度相同，方便后续处理。

填充的数量是由卷积核大小和步幅决定的。卷积核大小指卷积核的宽度和高度，步幅指卷积核在输入张量上沿宽度和高度方向的移动步数。

例如，如果卷积核大小为 3x3，步幅为 1，则填充的数量为 (3-1)/1=1

<img src="C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20221229223655791.png" alt="image-20221229223655791" style="zoom:33%;" />



#### 步幅

卷积层的步幅是指在输入张量上使用卷积核进行卷积运算时，卷积核在宽度和高度方向上移动的步数。卷积层的步幅通常是一个超参数，可以根据实际情况进行调整。

步幅对卷积层的输出维度有很大影响，步幅越大，输出维度越小。例如，如果输入张量的大小为 7x7，卷积核大小为 3x3，步幅为 2，则卷积层的输出维度为 (7-3)/2+1=3x3。

卷积层的步幅还可以调整卷积层的感受野（receptive field）。感受野指卷积层所能感知的输入张量的大小。步幅越大，卷积层的感受野越小。

在设计卷积神经网络时，通常会在每个卷积层之间使用步幅为 2 的卷积层，这样可以使得网络在计算时需要较少的参数，并且可以降低过拟合的风险。

<img src="C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20221229225240538.png" alt="image-20221229225240538" style="zoom:33%;" />

***

***

### 池化层

**从卷积层到池化层**

卷积层是一种常用的神经网络层，通常用于图像分类、目标检测、语音识别等应用中。但是，卷积层也有一个缺点，就是它对输入张量的像素位置敏感。这意味着，当输入张量的像素位置发生改变时，卷积层的输出也会发生变化。这可能会影响模型的精度和泛化能力。

为了缓解卷积层的敏感性，可以使用以下几种方法：

1. 数据增强：在训练模型时，可以使用数据增强的方法来扩大训练数据集，使模型对像素位置的变化更加具有鲁棒性。
2. 批标准化：批标准化是一种常用的归一化方法，可以在训练模型时使用。批标准化可以缩小每个神经元的输入范围，使其输出更加稳定。
3. 池化层：池化层是一种常用的神经网络层，可以缩小输入张量的大小，同时保留输入张量重要的信息。池化层可以在卷积层的后面使用，有助于降低卷积层对像素位

***

最大池化层和平均池化层都是池化层的一种，主要作用是缩小输入张量的大小，同时保留输入张量重要的信息。

最大池化层通过在输入张量的每个子区域内取最大值作为输出，可以保留输入张量重要的特征。平均池化层通过在输入张量的每个子区域内求平均值作为输出，可以抑制噪声，使输入张量的信息更加平滑。

在卷积神经网络中，池化层通常用于卷积层的后面，与卷积层一起使用，可以有效减少模型的参数数量，防止过拟合。池化层还可以增强模型的精度和泛化能力，提高模型的鲁棒性。

缓解卷积层的敏感性

***

在PyTorch中，池化层的默认步长与池化核大小相同

***

#### 最大池化层

二维最大池化层是指将输入张量的每个通道分别按照池化大小划分为若干个子区域，在每个子区域内取最大值作为输出。

例如，对于输入张量 X，二维最大池化层的输出 Y 中的元素 y[i,j,c] 可以表示为：

y[i,j,c] = max(X[i:i+k, j:j+k, c])

其中 k 是池化层的池化大小，c 是输入张量的通道维。

二维最大池化层的输出维度与输入维度的关系可以通过以下公式计算：

out_height = (in_height + 2 * padding - kernel_size) / stride + 1 

out_width = (in_width + 2 * padding - kernel_size) / stride + 1

其中 in_height 和 in_width 分别表示输入张量的高度和宽度，padding 表示填充的数量，kernel_size 表示池化层的池化大小，stride 表示池化层的步幅。

***

#### 平均池化层

平均池化层是指将输入张量的每个通道分别按照池化大小划分为若干个子区域，在每个子区域内求平均值作为输出。

例如，对于输入张量 X，二维平均池化层的输出 Y 中的元素 y[i,j,c] 可以表示为：

$$
y[i,j,c] = sum(X[i:i+k, j:j+k, c]) / (k*k)
$$
其中 k 是池化层的池化大小，c 是输入张量的通道维。

二维平均池化层的输出维度与输入维度的关系可以通过以下公式计算：

$$
out·height = (in·height + 2 * padding - kernel·size) / stride + 1
$$

$$
out·width = (in·width + 2 * padding - kernel·size) / stride + 1
$$

其中 in_height 和 in_width 分别表示输入张量的高度和宽度，padding 表示填充的数量，kernel_size 表示池化层的池化大小，stride 表示池化层的步幅。

平均池化层和最大池化层的区别在于，平均池化层在每个子区域内取平均值作为输出，而最大池化层在每个子区域内取最大值作为输出。平均池化层和最大

## nn.Sequential.

nn.Sequential是pytorch中的一个神经网络模块，可以用来组合多个网络层。它可以将多个层组合在一起，并逐个将输入张量传递给这些层，依次传播。

**2.Sequential实现自定义卷积层**

**3.sequential实现拼接卷积层和激活函数**

## conv1d/conv2d

`nn.Conv2d` 是一个 2D 卷积层，它在两个空间维度上执行卷积，通常是图像或特征图的高度和宽度。它通常用于图像分类和对象检测等任务。

`nn.Conv1d` 是一个 1D 卷积层，它在单个维度上执行卷积，通常是序列的时间维度。它通常用于自然语言处理和音频处理等任务。

```
import torch.nn as nn

# 一个 2D 卷积层，接受灰度图像（1 个通道），大小为 32x32
conv2d = nn.Conv2d(1, 10, kernel_size=5)

# 一个 1D 卷积层，接受长度为 128 的序列
conv1d = nn.Conv1d(1, 10, kernel_size=5)
```



### conv2d

在PyTorch中，你可以使用nn.Conv2d类来定义和使用conv2d层。

例如，假设你有一个大小为32x32x3的输入图像，你想要使用32个大小为3x3的卷积核来执行卷积运算，步幅为1。你可以使用以下代码来定义conv2d层：

```
Copy codeimport torch
import torch.nn as nn

# Define the input tensor
input_tensor = torch.randn(1, 3, 32, 32)

# Define the conv2d layer
conv2d_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)

# Apply the conv2d layer to the input tensor
output_tensor = conv2d_layer(input_tensor)
```

在这个例子中，我们使用了32个大小为3x3的卷积核，并使用步幅为1的步幅进行卷积。输出张量的形状为[batch_size, 32, 32, 32]，其中batch_size表示批量大小，32表示输出图像的高度和宽度，32表示输出图像的深度。

您还可以使用其他参数，如填充方式，来自定义conv2d层的行为。你还可以使用激活函数来添加非线性能力，例如使用nn.ReLU。

在PyTorch中，nn.Conv2d类有以下常用参数：

- **in_channels**：输入图像的通道数，例如对于RGB图像，in_channels=3。
- **out_channels**：输出图像的通道数，例如对于使用32个大小为3x3的卷积核的卷积层，out_channels=32。
- **kernel_size**：卷积核的大小，例如对于使用3x3的卷积核的卷积层，kernel_size=3。
- **stride**：步幅，即每次移动滤波器的距离。
- **padding**：填充方式。可以设置为'valid'（不填充）或'same'（填充为使输出大小与输入大小相同）。
- **dilation**：扩张率，即卷积核中元素之间的间隔。
- **groups**：将输入分成的组数。



***

## **优化器**

在深度学习中，优化器是一种算法，用于更新神经网络的参数，使得模型在训练过程中不断改进。

优化器的作用是通过调整神经网络的参数来最小化损失函数。在每一次迭代中，优化器会根据当前的损失值和模型参数，计算出参数的更新量，然后对参数进行更新。

常用的优化器包括随机梯度下降（SGD）、Adam、RMSprop等。这些优化器都有自己的特点，在不同的情况下可能会有不同的效果。

优化器即优化算法是用来求取模型的最优解的，通过比较神经网络自己预测的输出与真实标签的差距，也就是Loss函数。为了找到最小的loss（也就是在神经网络训练的反向传播中，求得局部的最优解），通常采用的是梯度下降(Gradient Descent)的方法，而梯度下降，便是优化算法中的一种。总的来说可以分为三类，一类是梯度下降法（Gradient Descent），一类是动量优化法（Momentum），另外就是自适应学习率优化算法。
![img](https://img-blog.csdnimg.cn/44c54a2ed1d040478da47aba34a18030.png)



### 从GD到SGD

GD和SGD（随机梯度下降）是深度学习中最常用的优化器之一。它的基本思想是通过计算模型的损失函数对于每个参数的梯度来更新参数。公式如下：
$$
w_{t+1} = w_t - \eta \times \frac{\partial L}{\partial w}
$$

#### GD

假设要学习训练的模型参数为W，代价函数为J(W)，则代价函数关于模型参数的偏导数即相关梯度为ΔJ(W)，学习率为ηt，则使用梯度下降法更新参数为：

$$
w_{t+1} = w_t - \eta \times \frac{\partial L}{\partial w}
$$
其中，η为学习率、表示t时刻的模型参数。

        从表达式来看，模型参数的更新调整，与代价函数关于模型参数的梯度有关，即沿着梯度的方向不断减小模型参数，从而最小化代价函数。
    
        基本策略可以理解为”在有限视距内寻找最快路径下山“，因此每走一步，参考当前位置最陡的方向(即梯度)进而迈出下一步。可以形象的表示为：

![img](https://img-blog.csdnimg.cn/c9d7590efc2b43259f7e5f203528679c.png)

#### SGD

 均匀地、随机选取其中一个样本，![(X^{(i)},Y^{(i)})](https://latex.codecogs.com/gif.latex?%28X%5E%7B%28i%29%7D%2CY%5E%7B%28i%29%7D%29)用它代表整体样本，即把它的值乘以N，就相当于获得了梯度的无偏估计值。



![W _ { t + 1 } = W _ { t } - \eta N\bigtriangleup J ( W _ { t },X^{(i)},Y^{(i)} )](https://latex.codecogs.com/gif.latex?W%20_%20%7B%20t%20&plus;%201%20%7D%20%3D%20W%20_%20%7B%20t%20%7D%20-%20%5Ceta%20N%5Cbigtriangleup%20J%20%28%20W%20_%20%7B%20t%20%7D%2CX%5E%7B%28i%29%7D%2CY%5E%7B%28i%29%7D%20%29)



​    基本策略可以理解为随机梯度下降像是一个**盲人下山**，不用每走一步计算一次梯度，但是他总能下到山底，只不过过程会显得扭扭曲曲。

优点：

        虽然SGD需要走很多步的样子，但是对梯度的要求很低（计算梯度快）。而对于引入噪声，大量的理论和实践工作证明，只要噪声不是特别大，SGD都能很好地收敛。
    
        应用大型数据集时，训练速度很快。比如每次从百万数据样本中，取几百个数据点，算一个SGD梯度，更新一下模型参数。相比于标准梯度下降法的遍历全部样本，每输入一个样本更新一次参数，要快得多。

缺点：

        SGD在随机选择梯度的同时会引入噪声，使得权值更新的方向不一定正确。此外，SGD也没能单独克服局部最优解的问题

补充原文链接：https://blog.csdn.net/caip12999203000/article/details/127455203

***







## 实例-Mnist

MNIST是用于机器学习训练和测试的手写数字数据集。它由 60,000 张训练图像和 10,000 张测试图像组成，每张图像的大小为 28x28 像素，并用 0 到 9 的相应数字进行了标记。

![MNIST 示例](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

***

## **leNet**

![image.png](data:image/png;base64,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LYmKiqKoqIhLly65ti0tLSU1NZXQ0NByNyRulk6no2nTply8eNH1pLi0tJRTp05V81UKcWNarbbc55HD4SAtLc2NEYm6IrfVhLjF5Ofnc+rUKbp06XJNpdTfstlsmM1m1xOZS5cucf78ebp163bNXXYhxK1n8ODB9OjRgxdffJGJEyeSnp7O22+/zYMPPkirVq344x//yLx589Dr9TRp0oS1a9fy66+/ct99990w6VUUhdjYWD788EPWrl3LokWL0Ol0BAcHM2nSJN5++210Oh1RUVF89913nDp1itdee63cPgwGAxEREXz55ZcEBQWRmprKihUrXMmOh4cHOp2OnTt3XlO1t1+/frRt25bp06czadIk7HY7r7zyCiNGjCAsLIwjR47U7JvZgP3444/861//olu3bly6dImlS5fy97//nYiICCZOnMjrr7+Or68vERERbN++nZSUFGbOnEmLFi148MEHmTt3LjabDU9PT/75z38SFRVF//79qxSLp6cnf/3rX5k9ezZarZbQ0FA2bNjgukEiRG1q2bIlPj4+vP322/Tr14+tW7eye/duRo8e7e7QRC2T6s1C3GKOHz/O1KlTufvuuwkKCqp03fj4eJYuXUr//v3R6XRs3bqV119/nZEjR94wYRZC1H9Go5EuXbqQkJDAzz//THJyMnfddRfjxo3Dz8+PyMhIvL292bRpE3FxcQDMnDmTbt26YbVaOX/+PIMHD3bNdcvJyaG4uJj+/fvj6emJv78/mZmZhIeHM3LkSHx8fIArRag0Gg0///wzR48exWazMW3aNHr27ImiKKSnp9OjRw+aNWtGWFgYJ06c4MiRI+Tm5jJkyBACAwO555570Gq1aLVa4uLicDgctG/fnl9//ZXBgwcTFhZGdHQ0SUlJ7Nq1i8OHDzNgwAAmTpxI48aNyc7OprS0lCFDhuDh4YGiKJw5c4bo6GhXpejbnd1uZ+PGjXTs2BGbzcZPP/1EXFwcTzzxBI888gje3t40b96coKAgtmzZwtGjRykoKGDmzJn06dMHvV5P27ZtMZvNbNu2jbi4OIKCgpg5cyZRUVFYrVbOnTvHwIED8fb25uLFiwwcOBCdTkdGRgbDhg1zfdekp6fj5eVFTEwMkZGRtG7dmj179rBnzx569epFaWkpUVFR3HnnnW5+10RDoqoqZ8+epUOHDkRGRhIQEICXlxc///wzcXFxeHl5MWzYMEJDQ+natSu5ubmEhYXRsWNHsrKyCA8Pp2PHju5+GaIGKKpMfhHihqxWKxaLBZ1O5+pdWMbpdGI2m9FqtRgMBtedalVVsdvt6HQ6zGYzTqcTo9GITqfD6XTicDjKPW0tW7/sZ6qqUlxcjFarxWg0uvZ78OBB/vrXv/L555/Tpk2bctsArnl8iqKwYcMG5s6dy/fff4+/vz+FhYVkZmbSqlUr17A0q9WK1WrFYDBcdz8AJSUlrrmAcideiPrHbDZTXFyMh4cH3t7e5eZH2u128vPzcTqdeHp6uhJXh8NBQUEBJpPJtX7ZPE6TyeSaA1pUVITT6cRkMpX7+y/b3m63YzQa8fHxcS0vKipyfaaoqkphYSEWiwWDwYDBYMBsNuPv7++K7+rPuqKiIkwmk+szqqSkhOLiYhRFwdfX1zW/1GKxUFpaiq+vr+u4BQUFeHh4XPM5fbsqLS3lmWeeoVWrVkydOpXS0lI0Gg3+/v7XzAPPz8/H4XBgMBjKzdeGK+dFQUEBcGV+cFki63Q6yc/Pd50bhYWFmEwmnE4nRUVF+Pn5uc6jkpISHA4Hnp6efP/993h5edG1a1dsNhtOp5Phw4czf/58YmNj6+jdEbeL/Px8DAaDa+663W6noKAAp9PpOpdVVcXLy4vS0lIURcFoNGI2m13/L259MrxZiEqoqkpSUhJffPEFcXFxNG3alNjYWIYOHYrRaCQnJ4cNGzawZcsW/P39iYmJYeTIkfj5+XH+/Hk2bNhAy5Yt+c9//kNJSQk9e/ZkypQppKSksGHDBiZPnuy6AD169Cj79+9n0qRJFBQUsHnzZtauXYufnx/9+/dn+PDhNGrUqFx8Z8+eZdOmTUyYMMF1kff9999jtVoJDAzk888/Jz09nXfeeYdRo0ZhNBrZs2cPYWFhGAwGTp8+zZo1a0hMTKRt27YMHjzY1Wdz3bp12O12cnNz2bJlCwEBATz66KMMGzaszn8PQojKeXp6Vpjo6XS6644K0Wq111Qz1ev15YoWAa7PqJvZ/nrblCWrVzMYDOXi8/Pzc/27LBkuc3WS9dt9XL0f4JrjiP/j6elZrkL21bRaLYGBgRVuq9frCQ4OvubnGo2m3DlQ9ru73rlxdXKRk5PDggULeOSRRzAYDHz//ff079+fAQMG/O7XJcSNXP35Alc+cyo636/+HJWbZw2LFLISohIpKSnMmDGDCxcu8OCDD2IymZgxYwabN29GVVVeffVVvv76a/r06UPbtm355JNPePPNN7HZbKSmpjJ37lyWL1/OnXfeSatWrViyZAnff/89Hh4erF69mr179wJX7qJ/++23xMfHU1xczIcffsg///lP+vfvT/fu3Vm5ciVvvfUWubm55eJLTk5m+fLlrqIwqqry008/sXXrVpxOp2uOlIeHBxqNhoSEBL788ktKSko4f/48U6dO5dKlS9xzzz2UlJTw4osvsm3bNgDXU+K4uDjuvfdeLl++zKxZsygsLKzbX4IQQogq0ev1TJw4kYceeqhKRadqg6IoPPDAA/z9738nPz+f9PR07r33Xl555RW5aSGEqDXypFeISmzZsgWz2czcuXMJCwvDZrPRqlUrzGYzx44d47///S9Lly5l8ODBAAwZMoRHH32UoUOHotfr8fX15ZlnnmHw4ME4HA5SU1PZu3cvsbGx9OnTh2+++YYhQ4aQk5PD/v37mTFjBgkJCaxYsYJPP/2Uvn37oigKffr04bHHHmP48OHXPAWpSPfu3RkzZgwJCQlMnz4df39/kpOTXcuXLVuGv78/r732Go0aNcJms/Hmm2/y/vvv06NHD5xOJ+3atWPevHkEBwcTExPD2LFjSUpKomvXrrXyfgshhKg5Go2Gnj17ujuMa/j4+DB8+HCGDh0KXHkyLFNnhBC1SZ70ClGJo0eP0rZtW9cwGA8PDyZNmsSYMWP45ZdfCA4OJjo6Go1Gg0ajoWPHjuh0Os6ePQtAcHAwLVq0QKPR4OHhgclkwmKxoCgKY8aMIT4+nuTkZBITE/Hw8KBDhw5cunQJb29vwsPD0Wg0KIpC06ZNCQ8PL9d65EYURXHd2b9eVczjx4/Ttm1b15BpDw8PBgwYQHp6Ounp6SiKQvPmzWnUqJGrB6dWq8VisdTEWyuEEOI2p9PppGqzEKJOSNIrRCXsdju/rfWWnp5OamoqGo3GVZDqt9uUJZsajabCIWVdunTBYDBw4sQJfvjhB7p3705ISAharRaHw+Fq6QFXioWUFcW62vUuFK7erjJarRa73V7uZzabDUVRXIVH6stwOCGEEEIIIapKkl4hKtGzZ0/279/Pr7/+ClypADhp0iRWr15Nv379yMnJYefOna4k9ZtvvgGuJLQ34uXlxX333ce6devYvHkzo0aNQqvVup4M792715VwHzlyhKSkJDp16lRuHwaDAZvNRk5ODgCZmZmuecJwJSn+bQJd5q677mL//v2up8dFRUV89tlnREZGXtMvUwghhBBCiFuVzOkVohLDhw9n/fr1zJs3j27dunH8+HEsFgsPPfQQoaGhjBo1iiVLlpCUlIRGo2Ht2rWMGjWKzp07c+DAgUr3rdFo6NOnD6tWrSIsLMyV0LZs2ZI//elPLFq0iKSkJPR6PevWreOhhx6iXbt2rqHTAK1bt6Z58+a89NJL9OzZk6NHj1JaWupa7u/vj91uZ86cOYwZM6bc8UeNGsV3333HzJkzGTx4MGfOnOHYsWO899570sNXCCGEEEI0GNKnV4hKqKpKRkYGGzduJDMzk5CQEO666y7CwsKAKz0hd+/eTVxcHAAdO3Zk8ODBmEwmsrOziYuLo3fv3nh7ewNw+PBhdDqd60lwcXEx+/btIygoqFxxqKKiIvbt28e+fftQVZXOnTszaNAg/Pz8yMnJ4ciRI/Tq1Qtvb2+Sk5P5f//v/1FSUkJ0dDRt2rRxbWOz2di3bx85OTl069YNvV7PmTNn6NevHzqdjpSUFL7//nsyMzPx8vJi8ODBdOrUCa1Wy8GDB9Hr9eVi/fnnn+nRo0elrS2EEEKI201xcfE1U4aEEPWHJL1CCCFEA5WYmMj+/fuxWq3uDuWmBAcHM2TIkAp7A4vak5OTw3//+19sNlutHqesFkVDu/wsK/Lo4eHh5khEbQgODkav1zNkyBD5Hd+iZHizEEII0UBt2rSJd999l+DgYHeHclNsNhtt2rShffv27g7ltnPmzBlmzJhBaGhorRYxtFgsmM1mHnrooQYzlUZVVSwWCwaDQSpR1zNOp5Pt27fj7e1Nx44dq7SP9PR0MjIysFgsOJ1OYmNjazhKURck6RVCCCEaKIvFQnBwMD169HB3KDdl3759t8xT6YbGZrNhMpno0qULRqOx1o6Tl5fH5cuXefbZZ2ncuHGtHaculZaWYrfbZYRCPWSz2SgpKaFx48bX1Da5WYcPH2bFihVMnTqVZcuW0axZM7p161bDkYraJkmvEEII0YBd3YZMiMqUnSu1eb6U9Z8v69F7q1NVFYfDgbe3d4N4PQ2Nqqquc7qqv5+ykQ/9+vUjMTGRxYsXs2zZMjw9PWsyVFHL5FtQCCGEEEKIKrDb7SiKIvM8bwMGg4GnnnoKm83GsmXLrtsOUtRfkvQKIYQQQghRBaWlpRgMBneHIepIYGAgS5YsYdOmTXzzzTc4HA53hyRukiS9QgghhBBC/E4OhwOHw4Fer3d3KKIOBQUFMWXKFD7//HOSk5PdHY64SZL0CiGEEEII8TtJxebb1z333EOfPn149dVXKS4udnc44iZI0iuEEEIIIcTvoKoqVqtVhjbfpgwGA1OnTsVoNPL+++9TWlrq7pDEDUiZOXFLys7O5oMPPiAhIcHdoYha1qNHD2bOnOnuMIRosE6cOEFaWpq7wwDg8uXLvPjii/j7+193uaIoDB06lHHjxkmyUcdUVSUlJYWzZ8+iqmqV92O1WrHZbDz99NMV/g61Wi1jx45lxIgR9fYpqsViQafTSWX025iXlxfPPvss06dPp3v37owYMcLdIYlKSNIrbknZ2dls3bqVtm3bSsXEBuzChQts2rRJkl4hatHFixdp3rw5oaGh7g7lhjIzM9m+fTujR4+WpLeOqarKpUuX8PPzIyoqqlaPk5CQwKFDhxg2bJirXUx9oqoqFosFb29vd4ci3Kxt27ZMmTKFt956i7Zt2xIREeHukEQFJOkVtySn04miKLRo0UIufBowq9XKxYsX3R2GEA2aoigEBwcTHh7u7lBuqOwJY3WeNIrqCQgIqNVzxel0kpmZidPprLe/57I2RdKXVyiKwvDhw0lISOC1117jnXfeISAgwN1hieuQMRlCCCGEEELcJGlTJK5mMBiYNGkSxcXFrFq1CpvN5u6QxHVI0iuEEEIIIcRNkDZF4npMJhOvvvoqmzdvZuPGje4OR1yHJL1CCCGEEELcBGlTJCrSunVrJkyYwNKlS0lPT3d3OOI3JOkVQghxy3A4HKSlpWG1WnE6na7/F0KI2iZtikRlNBoNsbGxDBo0iDlz5pCXl1ftfZZ9z8XFxWE2mytdt7S0lIyMDBwOB2azmYyMjHo7L94dJOkVQghxy0hPT2fYsGGcPHmSzMxMhg4dyrFjx9wdlhDiNiBtisSN6PV6pk+fTklJCcuWLcNisVRrfwkJCTz66KM899xznDhxotJ1d+7cyejRo8nMzGTLli08/PDDFBcXV+v4DYn81QohhLhlXH3XOiAggMWLF9OmTRs3RiSEuB2UtSkyGo3uDkXUc0ajkeeee46dO3dy8ODBau0rPj4eo9HIBx98QOfOnWsowtuT1FoXDVZqaipff/01vr6+7g5FVJHT6cTpdFbaE9JqtTJr1iwmTpxYh5GJyvzwww/AlT7LiYmJREZG0qNHD9q3b49Go8FqtbJv3z62b9+OxWIhNDSUUaNGufrEFhcXs3XrVg4fPozT6aR9+/aMHDkSk8lU7jiqqlJQUIDD4SA3N5edO3cSERHBhg0byM3NpWfPnsTGxuLp6YndbicuLo7169cTFBTEnXfeSW5uLi1atKBDhw51/h7dSsxmM9988w2qqrq9RUtZG5s77rijwjmVpaWlvPPOO4wZM6aOoxM2m41169ZRXFxcrXPF4XBw5swZVq1add3lqqpiNpvZvn077dq1q/Jxfg+bzSZtisRN69atG1OmTOHVV19l6dKltGrV6nfvY+/evWzZsoWcnBy2bdtG48aNOXjwIJ07d3a1DcvOzmb37t0MGTKkpl9CgyN/uaLBKikpISQkhD/84Q/uDkXUop9++omMjAx3hyGu8uGHH3L06FF69uxJnz592LNnD0uWLOHDDz8kJiaG77//nvnz59O/f39atWrF9u3biYuL480336RRo0YsXbqU//znPzzwwAN4eXmxfPlykpKSePnll8sdJy8vj9mzZ7Nq1So8PT2ZOnUq7dq1Y8SIEZSUlPDSSy8BMHr0aH7++WdmzZpFu3btCAgIYOHChcTHxzNt2jRJem/A6XSi1+sZPHgwQUFB7g7nhnbu3ElWVpa7w7gtqapKaWkpw4cPJzg4uNaOY7Va2bBhAwUFBbV2jN8qK2AlxM1QFIV7772X48ePs2DBAt58883f/TeRm5tLTk4OZrOZtLQ0MjMzeffdd5k+fbor6U1OTmbBggXyPXYTJOkVDZper78lLtJE1Xl5ebk7BPEbTqeT1q1bM2/ePNq2bUteXh5Tpkxh+fLldO7cmUWLFjFw4EDmzJmDl5cXDz74IA888ADffPMNDz/8MCtXruT5559n7NixaLVaunfvzl//+lcGDx5MREREhcfNyclhzJgxPPHEE1gsFjIyMtixYwejR4/mgw8+ICYmhrlz52IymYiJiWHs2LF1+K7c2nQ6HX5+fgQEBLg7lBuS4afupShKrZ8rFoulTlsGSZsiURUajYannnqKGTNm8NVXXzF58uTfNR/83nvv5dKlS6xfv54XX3yRkpKSWoy24ZM5vUIIIWqUVqulf//+tG3bFgB/f38eeughTp8+TXx8PBcuXGDgwIGuGxYhISH06tWLgwcP8vPPP2MwGBg4cCA6nQ5FUejfvz+enp43nBsVHh5O9+7dATAYDDRt2pSSkhKKi4s5deoUAwYMwM/PD41GQ+/evV13yoUQojLSpkhUVaNGjZg5cyb/+c9/2Llzp7vDuRi9DIAAACAASURBVK1J0iuEEKLGabXacv/WaDSoqorT6URRlGsuHhVFKbf8t3fDtVotTqez0mNWVFXV4XCgqmq5IlhlxxJCiMpImyJRXR06dOD555/njTfeIDExsUb3LS2Jbp4kvUIIIWqUw+EgPj7eNdfaYrGwc+dOmjdvTrt27fD39+fo0aPYbDbgSiGOAwcOEB0dzZ133klxcTFHjhxxFS06ePAgBQUFdOzYsUrxmEwmQkNDOX78OBaLBafTybFjx7h06VKNvWYhRMNksVjw8PCQNkWiyhRFYcSIEQwcOJDFixeTk5NTpf3odDr0ej25ubkA2O12du/eLSMQbpLM6RVCCFGjNBoN+/fv59VXX2XIkCEcOnSIHTt2sHjxYvz9/Zk1axaLFy/Gw8ODVq1asXHjRoKDgxk7diyhoaE8/vjjfPTRR+Tm5uLl5cVnn31G//79GTp0aJUSVUVRmDZtGgsWLMBisbgqYJY9VRZCiOspa1Pk7e3t7lDELU6r1TJhwgSef/55li9fzowZM373PgICAujUqRMrV65Ep9ORnZ3Ntm3bpKL4TZJ3SQghRI3SaDTcf//9REREsHXrVlRV5bXXXmPgwIEA3HPPPRgMBjZu3Mi5c+do0qQJ06dPd7UsmjZtGmFhYRw4cACAQYMGMW7cOPR6Pb6+vvz5z3+mSZMmmEwmpk6dSlhYGDqdjmeeeYYmTZq44rjrrrtc1V1HjBiBv78/O3bs4MSJE4wePZr09HQ8PT3r+N0RQtwqpE2RqEmNGjXiueee49lnn6V379707dv3htt0794dHx8fvL298fT0ZMqUKXz++ef88MMPREVFMX/+fE6dOkVQUBBarZYnn3wSk8lEu3btmDBhggzLv4rO4XBw8uRJqQgmboqiKHTo0AEfHx93h1IjcnNzycvLc3cYohry8vK4ePEi+/btq3Ado9FImzZtGsx5eysICQlh2rRpOBwO10Vj2VNVg8HAiBEjuOuuu1BVFY1GU+6i0mQyMW7cOB5//HFUVUWr1brmCPv5+fHss8+6/j1x4kTXsMMJEyaUG4I4cOBA13ynNWvWoNFo+Otf/4qPjw/Hjx+nsLCQrl271sn70dCVlJSQk5NTL+ZJFxQUkJKSUulngl6vp3379lLp+SoajQaz2UxmZuZ1qxSrqkpxcXG1OyLY7XYyMjKqda7YbDbMZjMnTpyodE6jyWQiOjr6mhoDN8tiscg5ImpU165deeWVV3jttdd49913b9hnunPnznTq1Mn13dayZUtmzZqFw+FwfTfecccdaDQaAgMDadmyJRqNhqioKCIjI2VY/lV0eXl5/OUvf6F169ZVvuNdNj/KbrdXqxpmcXExiYmJDB06FA8Pjyrto6ioiN27d5OZmVnlOOqjXr16ERUVVeXts7Ky2L9/Pz169KjyPkpKSjhx4gSbN28mJiamyvupT3bu3Mm5c+dcT5jErUdRFBISEnjhhReuu9xut1NYWMg//vEPBgwYUMfR3d6uTlZ/S1GUSj/nNRpNhV/WV+/z6nV+u/7VBbP8/PxYsmQJP/zwA82aNePQoUOMGDGCbt263fTrERU7d+4cu3btIjQ01O0XWUajkT179rBnz54K18nKymLt2rV07ty5DiOr38LCwoiNjaW0tPS6y1VVrZEEsLCwkK+++oqIiIgqX+vBlRtr//znPytcbrVa0ev1bNiwoUo9g8vaFFUnRiGuZ8CAAfz000+89957LFq0CJPJVOG61yv8+Nvv1ut9D15vu9udzmazodVqmT9/PiEhIVXaicVi4c0336RRo0aMHTu2yl94iYmJvPzyyyxcuLDSE6AyKSkpTJo0qcElvY8++ih/+ctfqnwCHzhwgDlz5rBixYoqx5CcnMzkyZMbVKU4u91Oz549GTJkiLtDEbXEbDbz1VdfYbFY3B3KbeO1116rd3Pg7rvvPjp06EBGRgYJCQkMHjyYnj17VvkJkCjP4XDQvHlzHnjggVtiKOjKlSsxm83uDqNead68OYsWLcLhcFx3ud1uZ8GCBRw7dqxax3E4HAQFBXH//ffX6uibwsJC/vvf/7oK5v1epaWl0qZI1AqdTsekSZN4/vnn+fLLL3n66aflPKsDOrhyV8DX1xc/P78q7aSsf5m/v3+17vLm5+ej0+nw8/OrctJrMpluiS/c38vT0xM/P78q/1H4+PhgMBiq9USzpKSkwTVmVxQFvV7v6hcqGh5FUdBqtQ3qZk19V9aftz7x8PAgMjKSyMhI+vfv7+5wGiSdToenp+ct8R0sF5jXUhSl0ptVdru9xq4BtFotRqOxVufUVzXZhSsjGG02G76+vjUYkRD/JyQkhFdffZVJkyYRERHB3XffLZ9LtUwGegshhBBCCPG/rFartCkSta5169ZMnDiRJUuWcPbsWXeH0+DJX7MQQgghhBD8X5siqXor6kJsbCwxMTG8/fbbFc6nFzVDkl4hhBBCCCGQNkWibhmNRiZOnEhRURErV67Ebre7O6Sboqoq+fn5ZGdnV1gHoL6Rv2ghhBBCCCGQNkWi7jVq1IjZs2fzl7/8haioqHpfXPW7775j7dq1pKenY7fbadWqFU899VS97+wiT3qFEEIIIcRtT9oUCXeJjo7mhRdeYPHixZw+fdrd4VxXSUkJr7/+OrNmzSIgIIDJkyfzyiuv4O3tzZgxY1i7dq27Q6yUPOkVohJms5kjR46QkZHh7lBEFdlsNs6fP8/nn3/O3r17K1yvdevW/OEPf5CLnascPHiQTZs23VaVr7VaLWPGjKF169buDqXB+eWXX0hOTq4X51NmZiaffPIJmzdvrnCd9u3bExsbW6sVhm8lGo0GrVZLUlISJSUl113H4XBQVFRUreOoqsrFixc5ceIETqezyvuxWCykpaVV2gdVURT69evHwIED0Wq10qZIuNWwYcPYu3cvc+bMoX379jWyT19fX0aNGkXz5s2rva+DBw+yYsUKZs+ezdixY11/J3fccQc2m43PPvuMQYMGERAQgKqqnDp1iqysLLy8vOjYsaOrU0paWhoBAQGuz9bi4mIKCgpcrXNTU1NJTk4GrlybhYWFAVcqyCcmJpKZmYlOp6Ndu3YEBgbedPyS9ApRieLiYvbs2UOHDh1+1x+WqF9atmwJwOXLl6+7PC0tjR9//JGRI0dK0nuVTZs28dVXX9GmTRt3h1JnTp06RZs2bSTprQWHDx/G09OTiIgId4dCbGwsTqeT1NTU6y7Pz8/nwIED9OvXT5Le/6XRaHjsscdo2bJlhfMOLRYLH3zwQbWOo6oq586dIy0tjc6dO1erl3bLli3Jzc0lNzf3usuTkpLIzc2lT58+KIoibYqEW3l7exMbG8vkyZPp3r17lfejqirJycmsX78ejUZDeHg44eHh1bqZ43Q6OXDgAGFhYcTGxpbbl8lkYs6cOa4Et7S0lE8++YSvv/4aHx8fUlNT6dy5Mx999BGenp6uHsUDBw4E4KeffmLVqlWsWbOGw4cP88wzz+B0OtFoNHh7ezN37lx69+7N6tWref/999FqtSiKQmBgIKtWrSI4OPimXoMkvUJUQlVVdDodgwYNIjw83N3hiFpy8uRJNm7cWC+eQNUnNpuN8PBwevbs6e5Q6kxWVhZWq9XdYTRITqeTjh070qtXL3eHckPp6els27atWk8aG6IOHTrQoUOHCpcXFxfz5ZdfVvs4TqeTiIgIhgwZUqsFpYxGI3a7HVVVpU2RqBd8fHzw8/Nj7Nix1d7X+PHjGTNmDO+++y6KohAbG1vlquQ2m40jR47QvXt3Vz9vs9nMoUOHXOsYjUasVivnz5/n66+/ZuHChcTExPDLL7/w2GOPsXPnToYNG0ZiYiIFBQWu7fLy8khKSgJg8eLFREZGsnLlSpxOJy+++CLbt2+nefPmLF26lD//+c8888wzpKWlMWPGDLZu3cof//jHm3oNkvQKIYQQQojbVlnSW3YxL0RDEB4eTvv27QkICOC9994jPj6eiRMnEhoa+rv35XQ6MZvN6PV611Pe7Oxs5syZg9PpxOl0YrfbWbJkCc2bN+eNN94gPDyc06dPExcXh9lsvqmWTE2bNmXbtm18+OGH9O/fn8mTJ7ue+AYFBbFu3ToCAgK44447mD179u8amSFJrxBCCCGEuG1JmyLRUOn1eh588EG6dOnCvHnzGDduHC+99BL9+vX7XVXKdTodoaGhJCUlYbVa0ev1hIaG8sUXXwBw8eJFxowZg8ViAWDXrl3Mnz+f0NBQAgICbnrUzAsvvIC3tzfffPMNK1euRK/X8+STT/KnP/2J119/nc8++4z333/fNRXh1VdfpVmzZje1bxnDIYQQQgghblvSpkg0ZDqdjjZt2rBs2TL+9Kc/sWDBAl5//XVXsaib3ceAAQM4dOgQv/zyC3Cl8GNISAhNmjRhz549rili//rXv9i4cSOLFy9m+fLlzJ49m4CAAADX9IGr6wJkZWVhs9mwWq0kJiby9NNPs3nzZlavXk2fPn1YuHAhFy5cICcnh//v//v/+OGHH/j4448JCgrinXfeuenieZL0CiGEEEKI25LT6ZQ2ReK24O3tzbhx43jnnXe4ePEiTz/9NLt3776pbRVFYfDgwQwYMIDZs2dz4sQJAIqKivjkk09YsmSJK6EtK2jl6+uL3W5ny5YtZGdnY7fb0Wg0GI1GDh48SGlpKYmJiaxfv971hHjFihUsWbIEi8VCZGQkHTp0QK/XU1RUxFtvvcVXX32Fp6cnnTp1IiIiwlVV/mbIOA4hhBBCCHFbcjgc0qZI3DY0Gg3du3dn2bJlbNiwgVmzZjFgwACefPJJWrVqVem2TZo04aOPPuKDDz5gypQpBAYGcvHiRXQ6Hc899xy//PIL3t7ePPzww+zatYv777+f8PBwWrRoQceOHfn222/p1asXs2bNYsGCBRw6dAhfX1969+7NuXPn0Ov1zJw5kxkzZvDQQw9hNpsJCgpi4cKFdOjQgSlTprBkyRJWr16N1WolJCSEt99++6Yr7EvSK4QQQgghbktOp7PKFW2FuFUZDAZGjRpFZGQkS5cu5dlnn+WZZ55h8ODBlY56MBqNzJgxg4cffpi0tDR0Oh3NmzcnJCSEgoICvL290ev1bNiwgYSEBDw8PIiOjiY7O5ucnBwaN27MqFGj6Nq1K5cuXSI8PJwmTZqQn58PQFRUFGvWrOGXX37BarXSoUMHV7Gq+++/n759+3Lq1Cl8fHyIjo7+ffOSL1++THZ2Nvfff3+Vh3aUNRLX6/V89tlnVdoHXCl9nZyczNChQ6tcMt5isZCUlFTtsvN2ux2Hw1Hl7Wva4sWL+fe//13lO5H5+fkkJSUxdOjQKsdQUlJCQkJClbdvqFJTU1m+fPlNVaUT9ZPZbCYvL4977723wmEyDoeDCRMm8MQTT9RxdPWXqqps2rSJ9PT0BvOUJC8vjzfeeIOPP/64wnUiIyOZO3cuLVq0qMPIqsbLy4uMjAw2b95c4ToV9a92hyVLllBQUOD288lms5GXl8eoUaPQ6/UVrjdp0qSbbpdxO/Dw8MBoNLJ+/foKi0LZbLZqJ5k2m41169Zx+vTpap0rhYWFqKrKqVOnKrxm1Gg0LFq0iDvuuKPKxxGivurcuTNvvPEG27ZtY968efz8889MmzaNwMDACrfRaDS0adOGNm3alPt52bxdAH9/f2JiYlz/bt68Oc2bN3f9OyoqiqioKNe/r35a6+XlVeHfW2BgIP369bv5F3gV3eXLlzGbza7+SNWVkpJS7X0cPHiw2vuYP38+3bp1q/L2X3zxBWvWrKl2HDXl0UcfZfTo0VX+cD927Bjvv/8+77//fpVjOHv2LM8880yVt2+oLl++jM1m4+WXX3Z3KKIWbd68mbi4OEl6f+Py5cv06tXrhsOiGgqr1crOnTvJzMy8JZLe8ePHM2zYsEp7D0+ePLkOI6pcZmYmY8aMKXdxVF99//33rnlt4gq9Xs+///1vsrKyKlwnNTWVZ599tlrHsdvtJCcnc//999fqZ4/FYmHNmjVcuHBBkl7RYPn7+zN69GjuvPNO3nnnHf70pz8xffp0+vXrh5eXl7vDqzENdnhzr169uPvuu6u8/YEDB2owmuoLDQ2lY8eOVU56CwsL8fb2pmPHjtWKo7I73rczg8FA69at3R2GqEXBwcHYbDZ3h1HvaLVaAgMDCQkJcXcodcJisdx00Yz6wNPTk8jIyErXqU8XNR4eHoSEhNwSNxQqexJyO2vcuDGNGzeucHlNXkc0bdq0Vs+V0tJSTCZTre1fuEd6ejpnz551VRu+Hg8PDyIjIwkODq7DyNyrZcuWvPXWW3z77bcsXryYH374gZdffhk/Pz93h1YjGmzSK4QQQgghhBBXO3LkCB9++GGlvWNNJhNTp06t8lDaW5W3tzePP/44vXv35h//+AdPPPEEf/vb3+jbt+8t39ZLkl4hhBBCCCHEbeHuu++mX79+lT7pVRSlXo2CqUtlc3bfffddPv/8c9577z127tzJhAkTCAsLc3d4VSZJrxBCCCGEEOK2UFBQwIULFypNenU6HS1atGgwQ3urQlEU/vjHP3LnnXfy0UcfMXnyZGbMmEH//v2rVSzYXSTpFUIIIYQQQtwW9u/ff8Phzb6+vkydOpW+ffvWYWTXKi0tZfv27aSlpVW6XnBwMA8++GCNH1+r1RIdHc3bb7/N5s2bWbBgAX379mX8+PE0a9bM7ZX2fw9JeoUQQgghhBC3hUGDBtGlS5dKn/RqtdpyLXjcxWazcfbsWRITEytdLzw8vFbj0Ol0xMbG0q5dO5YuXcrzzz/P2LFjiY2NrdXj1iRJeoWoAzabTfr43uJKS0ux2+2uBurXo9Fo8PT0rLA/5e3I6XRitVorvbi4lVgsFhwOB0VFRZWeC3q9vlzfwfquss8oi8VSb35/DoeD0tLSehGPxWLBYrFUeh7AlYI4t+JQwNqiKAoOhwOz2Xzdz0qn04nNZqv256iqqpSUlFTrXLFYLNhsNkpKSm742S9Vnm8d3t7eeHt7A5CQkEBeXt416+h0Ojw9Pd3+Oe7j48P48eOx2+2VrldXXQVat27NnDlz2Lp1Kx9++CGnTp3iiSeeqNEiV7V1LSVXZkLUgbVr17Jq1arbpq1LQzZkyJAKlxUWFvLee+9x33331WFE9duvv/7K6tWrMRgMDepmQGU9y4uKihg0aBDLli27JQqh9OrVi59++on4+PjrLq9PNy1Onz7Ne++9R1BQUL1pHfXjjz9WuCw5OZkdO3bQpUuXOoyofvPz8yMiIoKdO3dWuM7Fixfp3LlztY5TVFTEM888Q5MmTfDw8KjWvhYtWsSiRYuuu6zshtH+/fvrxZNB8fv8+9//5tixY65/2+12UlJSCAwMZMGCBfTq1cuN0V258bJ7924yMjIqXS8wMJAHHnigTmIymUyMGjWKoUOHEhsby0cffUSTJk1qbP95eXm888473HvvvTW2T5CkV4g6UVJSwpAhQxg7dqy7QxG16N1336WoqMjdYdQrdrud4OBgHnnkEXx9fd0dTp04efIk2dnZOBwOd4dyU1544QUmTpxY4fLs7GweffTROoyoYna7ncjISKZMmVLvn6ypqsrLL79MYWGhu0OpV4KDg1m9ejVWq/W6y1VV5dNPPyUuLq5ax7Hb7QQGBvLSSy8RFBRUrX1VJicnhwULFshorlvU3/72t3LnotVq5eTJk6xatapWz5ubZbPZOHXq1A2HN4eFhdVZ0lvG39+fnj174unpyejRo2tsv/Pnz6+VaylJeoWoI15eXjRq1MjdYYhaVN2nCQ2VTqfDz88Pf39/d4dSJ3x8fMjJyXF3GDfNy8ur0ifSGo2mXhUrMRgMBAUF1fuqqqqqotfr3R1GvaMoSqXJhNPprLEbZDqdjqCgoFr97lUUpUGNYrndXO97qVmzZmzZsoUTJ04QGRnphqj+j4+PD5MnT77hTVR3jnwxmUw0bty4xvZXW5+b8lcqhBBCCCGEEFwZUpyVlYXZbHZ3KCiK4ppXfOnSJbZt24bNZrtmvcDAQEaOHFnX4d1SJOkVQgghhBBC3Ha++uorkpOTXf9WVZWUlBQuXrxI27Zt3RjZtYqKioiLiyuXjKempnLu3DlGjhxZb5Le1NRUvvnmG8aPH1/uSXp8fDy7du1iwoQJbhkFI0mvEEIIIYQQ4rZjsViueaIbFRXFyy+/TPPmzd0U1fW1bt2ahQsXlissaLVaWbly5Q0LXdWFnJwczp49S1JSEjt27KBdu3blirvt3r2b06dPu63ehSS9QgghhBBCiNvC3r17cTgc9OvXj4iICIYMGUJYWJi7w7opv21/5unpSUxMDPPmzXNTRP8nMTGR//mf/yEnJ4fU1FQ+/vjjck90NRoN999/v9tqHUjSK4QQQgghhLgtHD16lLNnz9KpUyf+9a9/cd99911TDV5RFLy8vOpVkTKn03ndPuVnz56tF1Xiu3btyqJFi0hISOCTTz7hhRdeIDg42LVcURT8/PzcVnRLp9Vq61VVxpry448/Vqt65smTJ2swmuo7dOgQ/v7+Vf5dnTlzhszMTL766qsqx3DhwgWKi4urvL0QQlxPVlYWycnJOJ1Od4dSI1JSUsjOzubbb791FSC5npiYGFq0aFGHkVWNoihYLBbOnDlT4RC6Cxcu1Ju+ucnJySQlJbn9fFJVlcuXL7N9+3bS0tIqXK9x48b07t0bg8FQh9HVX4qioNVquXjxIvv377/ueWW1WsnNza32sbKysjh27Fi1zpXCwkLy8vJYt25dhX16FUUhIiKC7t27X/OkTtS9AQMGsHHjRh566CF+/fVX4uLieP/998ut4+vry3PPPceAAQPcFOW1Tp06xUsvvVTuWlxVVVRV5eGHH3ZjZFcYDAYaN26M0WjkiSeeQFEU8vLyyq1jsVgICwtzS+6pa9q0KSaTiZEjR2I0Gqu0E6fTyYkTJ/Dy8qJ169ZVfiH5+fkcOnSIAQMGVLn1h8Vi4eDBg5w/f/661c1uVnp6epW3vZqiKDz00ENVfj1Op5P4+HhSU1M5fPhwld/bX3/9lcLCQg4fPlyl7QFyc3Plw1oIUePi4uI4ePAg7dq1c3coNcJgMNCsWTM2bNhQ4TrHjx9n2rRpTJ48uQ4jq5qylhlZWVkVrlNUVERBQUEdRlWxTZs2ceLECaKiotwdCr179+bs2bOcPXv2ussdDgcpKSmsWbOG8PDwOo6uflIUhT59+pCeno7dbr/uOlqttkaewB07doyVK1fSrVu3at206d+/Pz///HOFyzMyMmjSpAkfffRRpe3BRN3o2LEj77//PmazmXnz5nHXXXfRt2/fcutotVqaNWvmpgivr2nTpvzlL38pl99oNBpCQkLo2rWrGyMrLysri4ULF153WefOnZk/f36Vc87q0Hl7e2MymVi4cCEhISFV2onFYuGNN94gLCyM8ePHVzkxSkhIYNq0aSxZsqTKTecvX77M3/72NyZNmkSPHj2qtA+40hh57969Vd6+jFarrdbrsdlszJ07l7Zt2zJ+/PgqJ7379+/n4sWLzJ8/v0rbAyQlJVUraRZCiOux2WxER0czbtw4d4dSZz777LN60Q7jZnh5eTF9+vRKi4+sXr2aFStW1GFUFbPZbPTp04dHH33U3aHckNls5sUXX6zWTfqGqFu3bnTs2PGaYZxliouLSUxMrPZxbDYb7du3Z8KECbV6EX7o0CF27drl9tEH4v+0bt0agFdeeYWQkJBa7eVcHZmZmbz11ls89dRT6HQ60tLSeOKJJ2qsl3VtaN68OcuXLy/3s6SkJD755BO6d+/u3jm9Go0GHx+fKidmer0evV6P0WjEZDJVOen19vZGp9NhMpmqHIvFYkGn07niqaqaGsOvKEq1Xo/NZkOv1+Ph4YHRaKxy0qvX69FoNNV6TwwGgzzpFULUCq1W67YvQneoL0OBb5ZOp6v0e7E+zXsDXNcB9Z27qpjWd4qiVPr7s9lsNXY9UvbZU5vnS1VH+4na17lzZ3eHUCmNRkNeXh7btm0DYNu2bTRq1OiapNfX15d+/fq5I8RreHh4EBoaWu5noaGhWCwW1q5dy8iRIyud+lNb6te3lBBCCCGEEEII/P39GT16NP/5z39ITU0lLS2N1atXX3MjJSIiot4kvRXx9fUlJydHWhYJIYQQQgghhLhCp9MxYsQIhg0bxrFjx1izZg0zZswgMDCw3Hr1qShxcXEx8fHx5X7mcDhYv349AQEBbhsZJEmvEEIIIYQQ4rZVWFjI6dOnsdvttGrViiZNmrg7pHI0Gg2RkZE89dRTBAQE1OspMllZWSxevLjcz7RaLW3atGHcuHFuKWIFkvQKIYQQQgghbhNpaWl8++23PPnkk3h6epKQkMA777xDYmIiqqri6+vLvHnz6Nq1a72qZePj40NkZKS7w7ihZs2asWjRItcwZq1Wi4eHByaTya3VyyXpFUKUo6oqxcXFWK3W372t0+nE4XBgt9tRFIWgoCDpPSlqhMViuWWqHd+M0tJSioqKyMzMrHAdDw+PavVnr2s2m43CwsIKh67VpwrFJSUllJaWujsMSktLsdvtXL58GR8fn+uuoygKvr6+8ll6HWU9cq8nPz+/wurPv0dpaSklJSXV2kdBQQEWi4WsrKwK96XVavHz86t3ReEaopycHH744QfGjh2Loij8z//8D2azmVdffRVfX1927tzJyy+/zNtvv02nTp3cHe4tpbCwkF27drFnzx6ys7NRFIXGjRvTt29ft885lr8sIUQ5WVlZ/Otf/yIlJeV3b3t10qvRaBg1ahTDhw+vhSjF7ea77767Zo7QrezixYskJiayb9++Ctdp2rQps2bNuiXu7Gu1WpKTk/nyyy8rTNLT0tLcftFTZtmyZSQnJ7s7DJxOJ+fOnWPmzJkVPgFRFIVHHnmEsWPHSkL0v7RaLRqNhtWrV+Pt7X3ddWw2zaiynQAAIABJREFUW7VvtNhsNjZs2MDOnTurlUDn5+eTk5PDpEmTKhyWqtVqef755xk0aFCVjyN+P4vFwrlz5/j73/9Onz59AOjUqRNHjx7l9OnTkvT+Dvn5+Sxbtoxvv/2W9u3bExERgdPpJCkpiS1btvDHP/6RJ598ssK/2domn55C1COqquJ0Osv953A4yv1X9jOn0+laX1VV138Adrud4uJicnNzXXfCCwoKKCoqorCwELPZjNlsxmKxuC4MHA4HqqpitVq5fDkHjYcBD70RrU6HRqNF0Vy5yNB6eOChN6LXe+Jh8MRDb0Bv8MLDYESvN6I3euKhN5J0fA/Z2Zfd/I6KhuLs2bN07tyZLl26uDuUOrNq1SrS0tJuiaR3+PDhfPnll5U+jX/33XfrMKLKHTp0iKCgIPz9/d0dCn379q1wmaqqpKenc+jQIR577DFJev+Xp6cnb731FsnJyeV63zqdTkpK/n/2zjw+jvK+/++9d7WH7vs+bMmy8Q3GYIzNDXFCIKUt5UpKk1D4kZSENDTkIikJaRJT0gQCLS1pCUcJhoSb2NxXOHxJtixZ97XaU3vfM/v7w5mpZCRhr2StbM/79ZrXjnZmnnk0uzvzfJ/v8YlgNpvx+/38+Mc/ntV5UqkU+/fvJxqNUlZWlnE7eXl51NbWTrtdFEX2799PV1eXYvTOE4IgEIlEMBqN5OfnT4q0UKvVmM3mjCLejgXpdBqPx4MgCJSWluL1ennrrbfo7OykoKCAjRs3Ul9fn9X7gyAIPPHEE2zfvp17772X5uZm9Hq9PK5sb2/nO9/5DkVFRfz1X/91VsLGlbungsIckE6nJxmQyWSSVCpFIpEgmUzidDoJhUK8++67xGKxSftJnlFRFOVj4vG4/BqNRgmFQkQiEdlYjcVictuS0atSqVCr1eh0OvR6PQaDAZPJhNFolF/1ej0Wi0UugqDRaNBqteh0OnQ6HQaDAb/fz6tvvM1pF19PafUicqx56I056PVGdAYTWt2RaRk+E/YpAzSFOUOr1dLS0rJgPIXzwRNPPDEn4ZnzQUFBAZs3b55xn0cffXSeevPJaDQaysrKFlyxmsNJp9OkUqlJk5oKh2hubqa5uXnSe9FolHQ6TU5ODg6Hg5/85Cdzcq7i4uIZjdbZIggCDodD/pyPl5SG4xWdTofT6eSuu+6ivr6edDrNq6++ypIlS0gmk3z00Ud0dnZy5ZVXZrurAOzbt4+f/OQnLFmyhJtuuokf/vCH7Nmzh8bGRhKJBL///e/5f//v/3HBBRdkrY+RSITnnnuOL3/5y6xevXrSd9hgMLBu3TquueYannvuOT772c9mJbdXGZEqnPQkk0n8fv8kQ1MyYF0uFx6Ph9dffx2TyUQ0GiUSiRCPx2XD0+PxMDo6yj/8wz98zCsrGbRer5dkMsn7779PKpUilUrJ29LptJzkbzAYMJvN5OTkYDabsdlsWK1WysrKMJvNGI1GLBYLJpOJnJwcDAbDIe/rn0O9pHa0Wi1arRa9Xi+3O/F9af+p6O7upv1AD82rzia3cGEPCBUUFBQUFOD/Jp+tVmu2u6KwwKmpqeFf/uVfGBgYoLOzk1QqRXt7O/F4nD/96U/88Ic/5KqrrmLVqlXZ7irxeJxnnnkGq9XKtddey0cffURbWxs///nPqaurI5VK8dvf/paHH344q0ZvMpkkFAqxePHiKSdt1Go1LS0tPPnkk4pOr8LJg2TsTTT8pvp7qnXptbu7G5/PR3t7O4Ds9Zy4eDwe3G43999//yRjdqJRm0gkUKlUPPbYY3L/JONRMg6j0SgvvPACOp0OtVo9yWjUarVEo1FEUUStVpOTkyMbrBaLRTZQd+7ciVqt5jOf+QwWi2XSfkajccocn8NvGlPdRJTZYAUFBQUFhUPjACmCSUFhJgRBYOnSpWzcuHFSFIVarWbNmjU8+uijlJWVLYgxViwWY3BwkMsvv5yqqirefvttWlpaJhnkZ599Ntu3b89iLw+NR3U63Yx59MlkMqu/T8XoVTgiRFEkkUggCAJer5fBwUHZ2xmLxeR1yaCc6v1oNDrJ+Dx8XQrVlcJ8J4b6xuNxudqmFOorCAIHDhz4WF9VKpX84Eun0zidTjl812w2k5ubi16vR6vVYrfbycvLo6GhQfaK6vV6eXssFuPtt9/m05/+NJWVlZO8plII8f79+3n55Zf55S9/Oe31k0KS169ffyw/JgUFBQUFhZOSeDyOyWTKdjcUjgPeeust2tra+MY3voFKpWJ0dJSuri42bNhAXl7egsj1lzCZTFRVVfH6669zzjnnUF9fz7Zt2xgZGaGyshJBEOju7kan02W1nwaDgbKyMt58802WL1/+sf6kUik++OADqqqq0OuPLE1urlGM3gWKNPM0sXCRlFA/sWjR4UWMDl8EQSAajTIwMEAgEOCNN94gEokQDoeJRCLyIpXkl3JGQ6GQHMobDofl9e7ubr761a9iMBhkw1TychqNxkle0MP/BynPNZlMTvK2wqHZNcnDOnGRvKXFxcVYrVbMZjNms5l4PM4rr7xCS0sLJpMJg8EgL5JB2t/fT09PD1dccYXctpT3Kq0/99xzVFdXc/7558t9mDiz53K52L9/P0uXLqW6unrKz8poNC6I2UAFBQUFBYWTESlVSKkjoXAkOBwODh48KP/d2dnJI488wtq1a7NuPB6OXq/nkksu4Qc/+AFf//rX2bx5M62trdx2222ceeaZBINBnnnmGa6++uqs9tNkMvFXf/VX/PSnP6WgoIDLLrtM/j0Gg0EefvhhduzYwT/+4z9mTX5NuTscY9LpNGNjY4yPj38s/HaqJZFIyB5Oyav50Ucf0dXVxcDAgGyURiKRSUbkxFBg6VjJw5pOp4nFYrjdbq677rpJ+0oGq0ajkT2cUu6nZBhK/4dUXMFgMGC1WifJ00w83mAwTFo3Go3k5ORMMlotFsukV7PZPGm7lNc6nTG5f/9+uru7OfPMM6f98fj9foxG44yFSqQc14V2k1NQUFBQUFA4MmKxGAaDQZmAVsiIVColF0FbiKxatYp//dd/5b777uORRx5hfHwcURR5/vnnqa6u5sYbb2TLli1Z7aNKpeKCCy5AFEX+/d//nccff5zS0lJEUWRwcJD8/Hy+/vWvs3Hjxqz1UQvIOZKBQCCjRuLxOB6PB71ez4EDBzIuQ93b20skEqGzs3NakfZPQpJo6e/vx2azZdQGgNvtzvjYiQiCwJe//GW5ZLdkqErM9APT6XRoNBqGhoYwm80EAgG5OJLUzuGSNpIRO3EBZA+mwWCQjTytViu/L+WnGo3GSdV+pXWDwUA4HMbj8fC3f/u3LF++fJKBO/FVWiTDV8mvUVBQONYIgoDH4yEcDme7K3NGLBZjaGiIzs7OafcpLCykqKhoHns1O8bHxxkdHZ12+0T5mWwiiiLhcHjS8zobpNNpwuEwPp+Prq6uacN3NRoNdXV1J6WnU0qJmkr7Mx6PMzQ0RCgUmvJYl8s1J30IhUKz0gQWBIFYLIbD4aCzs3Na491oNB7TKtIKCw9BEHA6nZSUlHDnnXfi8Xjwer2IokhOTg4VFRVZCxc+HJVKxcUXX8zixYvp6upiaGgIv9/Pli1bWLt2LXV1dVntn9bv9+P3+7npppswGo0ZNSKKIqOjo+j1ep577rmMZ9qkMNwvfelLGRtKqVSK/v5+vve972VsOAMMDg6i0ujQ6AyAGlQqMvqvVCp2tXdNWZRI9ec2VSrVofYnLvxfkSKfPySHNktG6nQhzoBsAE8kmUxOMrzj8bh8LqnNmdZVKhWxWAyv18sZZ5zB6aefnsnVUFBQUDgmOJ1Obr/9diKRyAkz8A+Hw9x5553T/j/JZJLVq1dzzz33UFJSMs+9O3pqamp46qmn2LVr17T7LBRdzFAoxLPPPoter8+KnuREBEGgp6eHd999d8rxlVS1+De/+Q3nn39+FnqYXeLxOHq9/mPXJicnB6PRyC9+8Ytpx6V+v59169bN6vyxWIyXX35ZdjpkgpQKdv/99/PQQw9NuY8oisRiMTo6OsjPz59NlxVAVt+AQ+G3yWRSVtqAQ2Nwi8WSdYNyfHyc73//+1x//fWcdtpplJSU8Pbbb1NTU0Nra2tW+zYdjY2NNDY2yo45rVa7IKIwtMlkEoPBwI9//GNaWloyaiQej/Nv//ZvlJeXc+WVV2b8o+/u7ua73/0u//7v/z7ljN2RMD4+zre+9S1uuOEGVqxYkVEbAP/0T//ES2+1UdB4OmqNDpVaAyoNR/uZqVRqrrzySvT6zMJnU8kET/7nXVxyZitXXHEFJpNJriJ8NOzatYuf//znPPzwwxn1A6Crq4svfelLGR+voKCgcKwQBAGLxcIXv/hFqqqqst2deaG7u5vXX399Vh6m+eRb3/oW119//YwRTp/+9KfnsUfTI4oiVquVjRs3ZjwemS9EUeSNN97IOFrveGYmmSKr1crzzz9PNBqd9vhnnnmGd999d1Z9kKITLrjggmOqPRqNRvnjH/9IOBxWjN45YO/evXzlK18BwOPxMDIywm233SY73SwWC3/3d3/Hqaeems1uIoqibJRLvPjii6xfv541a9ZksWefzMSI04WAFg51qrq6mqampowaicfj5OfnU1JSQlNTU8b/YDKZxGQy0dDQkLHOmtvtxmKxzOr/AbDZbGgNFkx5FYcM3gxRq9U0LVmRcdJ2MhHHmltAfX09a9asyXimxOVyYTKZZnVNYrGYkvuqoKCwYNHr9ZSXl09bdO5EIxQKLagBxSdhNptpaGiYcZ+F9P/odDpyc3MXvO6rKIpZKwyTbT5JpqiiomLG42eq+XE0qNVqcnNzZxVh+ElIqhIKs2ft2rV86UtfmjGdwmQyUVhYOI+9UjjWKL8eBQUFBQUFBQWF4w5FpkghE5YuXcrSpUuz3Q2FeUYxehUUFBQUFBQUFI4rFJkihZOFdDqNw+FgYGAAgEgkwvj4uPw3/J9O7kLjzTffJJFIsGnTpqwXtlXuFAoKCgoKCgoKCscVikyRwsmASqVCo9Fw9913ywWHXS4XBw8e5Pnnn5f3a2pq4r777stWN6fl/fffJxKJsGHDBsXoVVBQUFBQUFBQUDhSZpIpUlA4kbDZbNx88814vd5P3E9hZhSjV0FBQUFBQUFB4bhhOpkiBYVMEQQBr9dLfn7+ggqZNxgMs5bVUjjEwvlUFRQU5pV0Ok0ymSSRSBCPx+Wlq6sLj8vB3neex2TJzaxtUWCkdz8r6hQ9ZwUFhSMnGo3y/vvvT8pVm8jg4OCMFVfnE7/fj91uz3p/RFHE5/Px5ptvEovFpt2vsLCQc88997hXYZhJpuhoUavV9Pf38+KLL055XRKJBENDQxQXF8/qPPF4nL6+vll9V+LxOJFIhKeeeoqCgoJp92tsbGTdunXKhMBR4nK5uOWWW7jzzjs/scq8wpFz8cUXk0qlFsR9RzF6FRROAlKpFH6/H5/Ph8fjkV/dbjcejweXy0U8HpdFxEuK8vH3vEFgiodmLBYjEAjg9/sJBoMEAgFSqRQAGo0Gq9WKzWajqKiIiory+f5XFRQUjmO2bNmCw+FgcHBwyu0OhwO9Xj/PvZqa7u5u9u7dO2eyN7MhLy+PHTt2sGPHjim3S+HAzz//PI2NjfPcu7nlk2SKjobm5mYWLVpEV1fXlNtTqRQ+n2/WRq/T6eS1116jqqpqVrJclZWVPPDAA9Nuj8fjVFZW8txzzx1TzeATEUEQ8Pv9CIKQ7a6cULS2tma7CzKK0augcAKRTqcJh8PY7XaGhoYYGhpidHSUkZERQqEQkUgEQRDIzc0lPz+fiooKFi9ezLnnnovNZiMnJwez2SzrAQaDQUZGRujr62NwcJChoSHcbjfRaJRkMklpaSnLli2jtraW6upqysrKsFgsGI1GLBYLRUVF2b4kCgoKxxE/+tGPSCQS025/8803F0yxlnQ6TVNTE5s2bVrwXrVUKsWzzz4rT1Aez8ylTNEpp5zCr371q2k9sJFIhG9961s4HI5ZnUcURSoqKrjkkkuO6aSNx+Oho6NDMdwUFKZAMXoVFI4DRFEkGo0SjUZl4zUUCuF0OmUD1+124/P5iMVimEwmzGYzRUVFlJeXc/7551NVVUVFRQUlJSUAchuRSIRgMIjH42Hv3r309fXhcrlwu93E43FycnLIzc2lvLycU089lYaGBqqqqqisrJwyXGWhD/4UFBQWLmazecbiRBaLZR5788nodDpycnIW/H0vmUwu+D4eCclkck5lijQaDbm506fxGAyGOTNStVotJpMJg8EwJ+1NRTgcPmZtn+jo9XpaWlqOG93noaEhEokEDQ0NJ8Rvez5QjF4FhQVGNBrF6/XicrlwOp243W5cLhder5fx8XGCwSBqtVp+WBcWFlJXV8eaNWvIz8+nqKiI3NxcCgoKMJlMCIKAx+PB4XDQ0dHBjh07cLvd8hKPx9FoNJhMJoqLiykvL2fZsmUUFRVRVFREfn4+xcXFC6qwg4KCgoLCyUc8HldkihSOCYWFhXz3u9+dk1zxY40gCPz+97/H6XTy7W9/e8GkfCx0lFGsgsI8kE6nEQSBeDxOKpVCEARSqRRut1v21I6OjuJwOHC73QiCgCAImEwmCgsLKS0tpbm5mcrKSioqKjAYDBgMBnQ6nZzbJIoiyWQSu91OW1ubHI7scDiIRqOkUin0ej1lZWWUlJSwYsUK6urqyM/PJycnB4PBgNFoxGg0Zl1LTUFBQUFBYSKKTJHCsUStVpOXl5ftbhzXpNNpAoEAb7/9Nm+99RbhcBiVSkVxcTGtra1s2LCBgoKCrI0xFaNXQWGOSSaTBINBgsGgXOypq6sLl8vF1q1bcTgchMNhwuEwOp0Oq9VKbm4upaWlnHrqqVRVVVFaWkpJSYkcdpVOp0kkEgQCAbxeLw6HA5/Ph8PhYGxsjJGREYLBIOFwGKPRKHt66+rqWL9+PTU1NVRUVJCXlzerIhoKCgoKCgrZIBaLKTJFCgoLmLa2Nn7wgx8AUFFRgUqlIp1O43A42LdvHw888AC33nor55xzTlZ+x4rRq6CQIdKM1tjYGB6Ph8cee0w2QqWqxoIgkJOTg9frxWq1UlpayooVK8jNzZWNXanasRQ+nE6nicVi2O129uzZw+joKMPDwzidTgKBAIFAAJ1Oh8VikcORzz33XPLz87HZbPKrxWJRDFwFBQUFheMeaeL3eAg9VVA41qjVapYuXUpNTc2CicxLJpM89dRTNDQ0cPPNN1NdXT1pu9Pp5Ne//jWPP/44Z555Jkajcd77KBu9UlGbTIjH47LWZygUynigLVWWDYVCGc8AhMNhUqnUrP4fOPThzQXSjTrT/yeZSCCKonxtM21n4rXNlEgkknU9wvlAChNOJBL4fD7i8TgDAwMMDw8zNjaG3W7H4/Hg9XplbdtAIMCHH35IaWkpy5cvp6KigsrKSgoLC9FoNDz22GMkk0muueYaOcw5FovJxq2kSznRa6tWq9Hr9eTn51NdXS17gevr6zEajWi1WnnRaDTK7LfCMSGVShGPx6fclk6nj6t7gnQvTafTR32s9L+KoohGo8FsNiu/uSyRSqWIRqPT6j5KxY4WAlIqS7ZJpVJyQcSZih3p9foFoad5OIlEQn7WZYNUKjXtuHCuvm9SWtNs2kr8ecwYiUSmHYurVCoMBsOCMZayRTqdJhQK4fP5SCaTcgqYRqPBYrGQn5+f7S5Oi0qlYuPGjaTT6QXzOSaTSUZHR/nMZz7zMYMXoKSkhAsuuIC77747a/dErVqtJh6Ps3XrVgoLCzNqRBAEPvzwQywWCx0dHRkPBMbHx+nt7eXb3/52xjfdWCxGe3s7v/jFL+QqtZnwwQcfALPPG0mn07zyyisZfylFIYXL6eLJJ5+ko6Mj4344HA56enq49dZbM25DKqR0IiHpsnV3d8uhyD6fT17Gx8cZGxvjkUceobCwUJb6OeWUUygoKKCoqIiRkRHeeustfvazn01qO51O4/f75UJUbrdb9gY7nU58Ph+pVAqVSkVeXt6fdW0rWLNmDSUlJXKI8/FSSVDhxEOr1dLV1UUgEJh2n2AwOI89mh0HDx7k+edfIBqNHPWxqVSKRCJBMpmkoKCAv/mbv5nywa5wbFGr1fT19fHrX/962uItbW1tVFZWznPPpmbXrl2MjIxkuxuIosjY2Bj/8i//Mm21YpVKxVlnncXnPve5BVcYZy5lio4GlUqFRqOhs7MTu90+5T7SJMxsEEWR3t5eOjo6ZmX0xmIxPB4Pt99++7TjaJ1Ox+c+9znOPvvsjM9zIvDuu+/y+OOP09bWRjKZRK1WU1RURDKZpKSkhGuvvZYNGzYs2Ii5hWLsSkiTBe+//z4bNmz42H0mHo+zb98+9Hp91q6p1mazYTAYWLJkCTabLaNGUqkUPT095OXl0dTUlLHR63A4MBqNNDQ0ZFzSPRwO8+GHH1JVVUVNTU1GbcChB5UjOjc6Z/n5+RlXvhVSSXR6PcXFZhobGzO+tjqdjra2NpqamjI6HsDtdmf8HZlPpAdGOp2etPh8PjweDy6XS66G3NPTg0ql4r333sNgMGC1WikoKKCkpISmpibUajU7duzg6quvpqGhQS70NPFziMViqFQq3G43Q0NDkzRtvV4v4XAYr9dLTk4Oer2euro6Nm7cSElJCTabTZbosFgs6HQ6xXuksGC44oorsFqtMw7C/uu//uuI2pJ+hzNtm+k80vZEIoHf78fn88mvoVBI1pN+6KGH0Ov1chSGNIMvCAJOp4v+/r5pNCxVaHV6dAYjeoMJnd6AzmBCb8iZsG5ErTXTt2s3F154oWL0ZoHW1la++c1vzjgR4/F45rFH05NOp2lra6OiomJBaJbX1NTIEUZT4Xa78fv9XHzxxQvK6JU8qdnwQBsMBq699lpaWlqm1b4NBoNHfB+cDlEU6evrIx6PU19fP6txQFNTE16vd8pt6XSagwcPUlpayoYNGxac4TRfRCIRvv/977No0SK+8Y1vUFhYKIfbejwenn76aX7yk59QXFzMkiVLstzb4wO9Xs8555zDnXfeycGDB1m3bh0VFRXY7XZ8Ph+9vb10dnbyhS98ISuhzQBarVZLTk4ON9xwAxUVFRk1Eo/H8fv91NTU8OUvfzljC76jo4P333+fr3zlKxnnbbjdbnbv3s3VV1/NaaedllEbUl8OOvZkfLyEWq1m3bp1GRvxyUScndv/m/PPX8+Xv/zljG+E7777Lu3t7bPy9HZ3d7N9+/aMjz8WSINZKVw4kUgQiUQYHx+nr6+PkZER7r77bkKhEBqNBoPBgNlspqCggKqqKhKJBE1NTWzZsmXKH6HL5eK9997DbDYTj8flAbbX68Vut8vG7ejoKDfccANms5n8/HwqKys59dRTqa+vp6qqihdffJFkMskNN9yQhaukoJAZra2t8mAvlUpNuTz++ON4vV6Gh4cRRRFBEOQwYFEUsdvt+P1+2tvb0Wq1JBIJeeAtaUT39/cTi8XYu3evvC2ZTMrhfnBocun999/HaDTKIZg6nQ69Xo9Wq5WjNoaGhrBYLGi1Wjn33WAwYDKZ0On1iHor6y++jsLSanKseRhNFvTGHAwmyxENABOxCI//9PoFGQJ6MlBSUsIXv/jFGfdxOBzz1JsjY82aNbOacJ4vuru7GRsbWzCh4RLxeDxrg2S1Ws0ZZ5zBGWecMe0+Y2NjbNu2bU7O19LSwoYNG47Z5Lf02YqiuOA+5/nk/fffx2Kx8KMf/WjKyIfNmzdzzTXXsHv3bsXoPUJUKhWf+tSnqK6u5uWXX2ZkZISPPvoIQRDQ6XSsWrWKG2+8kVWrVmXP05uVsyooZIAk7yP9kOx2Oy+99BLxeJxwOIwgCHIerNVqRaVSkZuby9lnn43VasVqtcoeVekHt23bNmw2m/xATaVSeL1eeZFyef/nf/6HRCKBIAhotVrMZjOFhYU0NDRQUFCAXq/nO9/5DgUFBeTn539s8KzRaOYsT1xB4UiQJoOi0eikHPKJi7RdMkAjkYj8vuQlTSQS8jKxfoO0rbOzk97eXjQajZz3Jk1GTRxg2e12WWprojyWlFtms9lYvny5vE2S4tJoNGi1Wl566SXWrl3LmjVr5GNNJpMst+V2u7n//vu5+eabqaurm/KaPPPss2x/7wCnbr58Pj8KBQWFDJEm3BSZIoW5RBAE+RkyHRaLZcFPDLS3t5NIJFi9enW2uyKzfPlyli9fnu1uTIli9CpklXQ6LQ+Sk8kkyWSSSCRCf38//f39dHd3Mzg4yPDwMMFgUPb6GI1GQqEQ9fX1FBYWUlBQQE5Ojuzd0Wq19PT00NHRwZo1a+QiNJIhID1IA4EAHR0deL1eOc9WMm6NRiMWiwWAVatWsXTpUqxWqzxo1+v1GAwG9uzZg91uZ/Hixdm8lApZRgq/lbQkpZl06b3D16fbnkwmCYfDshE6cX3iEg6H5aI00j7BYFDeLhU0meh5hUOzsRO9CJKBKhmph4cZq9VqjEYjZrNZnjySctAtFgsHDx6koqKCqqoq2YiVXnU6HaOjo+zcuZPLL7+c3Nxc1Go1Go1m0uv27dtJpVJce+21qNVq1Gr1xzwde/bsYdmyZWzYsGHK6x+JRJTUAAWFE4x4PK7IFCnMOcuXL2dkZITf/va3XHLJJZOiMVOpFM8//zw9PT18/vOfz2IvDyEphUxVUPLll18mEAhQVVWFTqdb0MW3FgKK0aswb0hForxeLx6PB5/Px9jYGENDQ7IkTygUIhaLkZubS15eHuXl5TQ3N3PeeefJocIVFRUcPHiQ66+/nrPOOutjoeNS3p9U8bqjo0MOSZaKcUWjURKJBKFQiIKCAnJzc2lqaqKwsJDS0lKKi4vJycnB7Xbz8MMPc/rppyv5eycokpE60Ut5+Pp07x2+/9DQEM899xw9PT1Eo1F5megplbylkld1otG3WvtXAAAgAElEQVQqPdimml1Wq9XodDo5VFeafNHpdLLxKIX7SrUapIkZyRA1m83k5ORgsVg+ti5FQUy1faYB586dOznllFOmDQFLJBKYTCZKSkrIy8ubch8pVFgJGVZQUJBQZIoUjhXFxcXceuut/Pd//zdPP/004XCYZDIpTwrbbDa+9KUvceqpp2a7q4RCIe644w6efPJJ6uvrJ+Xbu91uBEHgnXfeobGxkfvuuy9r/Uyn0wwODs5YHR7AbDZTU1Oj6PQqnBikUimGh4cZGhqiv79/ktyPz+cjGAzKRaNKSkqoqalhw4YNlJWVkZeXR35+PgUFBeTl5cme1ulIp9NEo1HGx8fxer2yUR0MBvF6vQQCAV544QWMRqNcbXnRokXy4P7NN9+kqamJCy+8UJlJPs6QpHQm5nNPDM2dal16TSQScthvIpGYMlc1mUwiiuIkz6PkBZU8o1J0ghTuGwwGOXjwoNwGHPKsSl7NicbpRMMyLy9PDrWSjFPpOyqF5Uvrer1+0iIZt5+0frIWLFFQUDg+ybZMkcKJzac+9SlWr17N4OAgbrdbfmZbrVYaGhpmXVBsrjCbzVx//fWy7OjnP/95uTDeY489RjAY5Itf/GLW8t4l4vE4Dz74IO+8886M+61YsYI777wzuzq9CgozIYVdCoKAx+Ohr6+PaDSK1+ult7eXvr4++vv7sdvtuFwuVCoVWq0Wm81GdXU1VVVVnHXWWdTV1VFXVyeHIk9cDr+5iKIoe8CkME6Px0N3dzfvvfeeXKk1lUrJhoTVaqWwsJCqqiqCwSCjo6Ncd911k0IpJ4ZOmkymKUMpFT7OxBDxiYWKJi7TbZv4vmQkSrmksVhs0rpkwE61TMxFTSQSk8KDDz+XWq2Wv1sGg0EePE38vCeGFU8MTZb6mEql5OOk3NKJuaY2m42cnBzMZjNGo5G2tjZWr17N2rVrsdlsWCwWOSR44iJVAJf6cvjrTNsUFBQUTgZisRg5OTnZ7obCCUoikWBsbIyxsTFqampobm6e5GgZGhoiNzc366olarWapUuX8pOf/IQnnniCJ554gr/+679m7dq1FBcXo9PpFkSxLaPRyA033MD4+Dgffvgh3/nOd6aM0sjPz89adXjF6FX4GIIgyNqyDocDh8OBy+XiwIEDDA8Pc8cdd2AymRAEAYPBQHFxMcXFxTQ3N3P22WdTXl5OVVUVZWVlFBQUHNFAXRRFWb/WbrczOjqKw+FgcHCQ0dFR/H4/yWQSnU5HaWmpXL31tNNOo6CgAJvNRm5u7qQZ4QMHDuByuZSH5gTS6TTj4+OMjIxM6d2UvJfS68T3JJ1SyaspGZ4T16Vc1pmMYmlCRKPRkE6n0Wq1k7ynEwsgTQzzlfQSpWJlUli7ZLhKExrSMtGQlcJ+J3o/pSrAE3NQpdfDQ4il9w9/7/AKhN/97nf5i7/4C/7qr/5q/j7ULCOKohxZMRWBQEDOJz5ZiMfjOByOKSfzjgZBECgsLFS0ujMgEAhMq6sKLJjCgul0mlAotCD6I9UEGBkZIRQKTbmPRqORB9rHEul6HC8pD9J9cLp7XSRy9NrgUxEOh6fM7TxSpOg4qdL9dHKaer2e4uLiBatROxf88pe/5OWXX0an0zE+Ps5FF13ErbfeKo8Zf/CDH/DZz36WT33qU1nu6SGMRiNXXXUVS5Ys4Xe/+x379+/H5XItiN+Iz+djdHQUgIsuuoiuri7q6uqmTWkaGxubsb3ZfMdnQjF6TxIO16yVCIVC9PT00N/fT1dXF319fQwMDDA+Pk44HCaVSlFUVERRURG5ubno9XquueYaTjvttEleLElj9pP6AIcGckNDQ3R3d3PgwAG6urro7+/H7XYTDAbRaDSUlpZSUVFBS0sL5557LqWlpeTl5WG1WrHZbAwNDXHw4EGWLl2asRzUyYggCNx99920tbUdMkIlg1QQEUXhsPcm/C2IiOlPNlzUKjVqjRqNZHxKBYvUf/ayy9s0qNWHwoWliQpRFGVZqYkG6sSKv5JX1WQyySG/0nuHRw5IhvUnrSse1NmRTqd58cUXpw1VSiQSFBQUzHOvssuePXu45557pt9BpeJIvnGCIGCz2aitraW1tZWmpiYWLVok63l/krbxyYpOp+OZZ57hjTfemHYfp9M5jz2aHkEQeOCBB6bVgJ1PpAnNa665ZtqQYrVazXe/+122bNlyzPuS7XDNI0Wn0yGKIi+99NK0RmIikchYFlQiGo3yhz/8gaGhoVm1EwwGaWtr49VXX5322afVannkkUdoaGiY1bkWKqOjozz77LP8wz/8AytWrGD37t386le/4pRTTuGyyy7LdvemRa1Ws3btWqqqqnj88cd55ZVXOP/887PWn/Hxcd58800cDgcvvvgigFzf5JZbbsk4NSEcDh8Tb7Bi9J6AiKJIKBQiFAoRCAQIBAJ89NFHjI6Ocscdd9Df38/w8DAej4doNCobrxUVFdTV1bF27VoWL15MfX09FRUV8pe2u7ub9vZ2Tj31VNatWzfluePxOMFgkPHxcfl1eHiYnp4eOjs7ZW+uVGynpKSEpqYmLr30UnlQV1xc/IlGyEwz+AozI4iweN0WFq3YgE5vQKszoNHo5t3wC3gdvPWHX/PFa/+SZcuWkZOTo1TpPA7Ztm3bjLqoe/bsyWpxjWyQSCQQdTaqm9cS8DrwexyEAh6SiThGkwVrfjEFJZUUlFaTY80nx5yLObcAgzEHler/Bs3b7v0GS5cuBWDHjh08+eSTBINBiouLKS8vx2az4fF42LdvH6IoUlBQoEi7ALfddhtXXXXVjIbkQonGEEWRRCLBlVdeSWFhYba7MyOiKPLyyy/T399/TM9zvMkUFRYW8uKLL+L3+6fd54MPPuDuu++e1XmSySR2u50rr7xyWg/aXBAOh9m2bRsej+eENXrb29upqanhs5/9LAD19fWYTCb+9V//leXLl9PY2JjlHs5MWVkZX/3qV7n00kvl6Lj5HDv5/X6efvppHn74Yf7yL/+Sc845Z06NVIPBQH19/Zy1J6EYvcc5sViM0dFR7HY7w8PDjIyMMDY2ht1ux+124/P5UKvVxGIxAoEA4+PjNDc3s2nTJoqKiigpKaGkpISioiLMZvNR/Wii0SjDw8OypNDAwACjo6M4nU5cLhfpdBqj0UhhYSGVlZVs2LCB4uJiSktLKS8vp6Sk5KTzAC0EdHo9ZQ1LaV55Vlb7YbLYMFvzKC4uUcrsH8dI95DpCIVC04bQnaioVCoKSqo474qbMOZYScSijLtHCXjG8LnteF3DeB3DdHz0Gsl4DI1GS15xBXlF5RSW1VJcWU9hSTUWi5X169ezdu1akskkHo9Hvr+PjY2xb98+Ojs7uf3228nNzaWqqoqGhgYWLVpEQ0MDLS0t5ObmZvtyzDtWq5XW1tYZ91lIEUJ6vZ6KigpKS0uz3ZUZEQRhXr5Px6NMUU1NzYzbPR7PnJ2rsrLymE6QBIPBEz4trKioSFYQKSsrA+C8885j+/btPPjgg3zzm9/Mcg8nk06n6erqYvfu3TQ3N9Pc3Mwbb7zBG2+8QTAY5PTTT+fSSy+dl4mi7u5u7rnnHvx+P7fddhvnnnvuMT/nXHFyjUSOE9LpNKJwKMcyHg2RSsTx+Xzs2bOH3t5e+vv76enpkWV+pBxMs9lMeXk5FRUVnH766dTV1VFdXY3ZbKa9vZ2f/exn/PjHP/7ESq5S0apkMinnbIZCId555x3cbjdbt24lHo/jdrvlYkJ5eXnU19dTW1vLOeecw+LFi7FarZhMJnk5XkKVFBQUFOYSvdFEaVUjpVWNf84zF0jGo8RjERKxKCGfm+G+fTiHe9j77gsEx12o1Rr8riEee+wxDh48SGtrK2VlZTQ0NNDc3IxarWb//v088cQTfOc73yGVSrF3717a29t55ZVXCIVC6PV6ysrKaGlpYc2aNSxdupSioqJJqQMKCgsJRaZIYT445ZRTyM3N5Stf+QpXXHEFn/70pzEajdx000386Ec/4vbbb5/TiYrZ8vLLL3PPPfdgMBhIJBKceeaZvPrqq5x33nnU1tby29/+lkAgwA033HDM+hAOh3n22Wd56KGHuOiii7jqqqsWfHTK4ShGb5ZJpZKEA+OEA+OEAl5C/kOL2zGE1zGMxzlCz/6d/LLjLZ5++mkKCgrIz8+nurqa888/n+rqaurr66muriY3N3fafBKn04lWq/3Y7J0oivj9frxeL06nE4/Hg8vlknN77XY7gUCAZDKJwWDA7/djNBrZvHkztbW1sqF7os8KKigoKMwFhwqyadHkWDHmHBrYF1fUUd+6Fjg06I9Fg7jtAzzxi6+h1+vZv38/r732GtFolJycHIqKiqiurkaj0RCNRkmlUqxcuZKzzz5bzvV1uVy0tbXR2dnJgQMHeOihh3C73Wi1WoqLi6mvr2fZsmVUVFRQXl5OZWUl+fn5J3ThGoWFjyJTpDAf6HQ6fvjDH/Lb3/6W1157jU2bNmE0Gqmvr+cHP/gB//Vf/4Xb7V4QkTLhcJjt27ezefNmvvGNb9DW1sbf//3fc9lll/G1r30NlUpFZWUl99577zExekVRpKOjgwcffBC73c73vvc91q1bd1xFYkgoRu88EgqO4x4dxOMcwjnch3tskHHXKEG/h0jQjyAImMwWTGYbRaXVFFfU03TKOqKxGJedv47rrruOgoICcnNz5UImR4NUIVkqWtXf309vby9OpxOv10s0GsVqtZKfn09jYyNr1qyRw2jy8/MJBoPcfPPN3HjjjZx++unH6CopKCgonLyoVCpMOTaqG0+hsKiEyy+/nBUrVuD1ehkfH8fj8ciV7Ts6Oujt7eWWW26hsrJSLv63dOlSFi9ezLnnnsu5554rV213Op2MjY0xOjpKd3c3L7zwAg6Hg3A4TFFREVVVVbS2trJs2TJaWlo+MWRTQWGuUWSKFOaLuro6/umf/olgMDgpsqC8vJyvf/3rjIyMzJi6M18kk0l8Pp9cRXrZsmWUlZVN0hEuKyuTdYbnmkcffZRHHnmETZs28c1vfnPBp2HMhGL0zpJ0Ok1aSCIKKdLin1+FBMlIgER0HCEa4J7bdhIcdxKPx9BqdWh1OnILyygqq6Fp2WmU1yyipLIea14xWp0OtebPeqIaLUIqxZ4P3qCmpobFixdPaehKci+RSIRwOEw0GiUYDOJ0Ojl48CAHDhygvb2dnTt3cuGFF8p6tjU1NTQ2NnLRRRfR1NRETU0Ner1+UvXbiZVtu7u7ldlXBQUFhXlGClMuKyuTKzYLgsC+ffvYtm0bt99+Ow6Hg/3797Nz506eeuopYrEYer2e6upqFi9ezKpVq6irq6O2tpalS5eyZcsWtFotyWSSYDDInj172Ldvn3y85FWuq6ujtbWV1atXU1lZic1mkyv2n2y52grHlmQyiUqlWhASLAonB2q1ekpvrsFgWDBFvEwmEw0NDbz66qvU1tYyODjI+Pg4b731Fhs2bECr1fLuu+/OqZ6wIAh0dXVx33334Xa7ueOOO1i5cuVxf88/vns/z4ipBKl4GCEROfQaD5OKh0nFgqTiIYRkjHRahDRodEY0hhx0RgutazdRVFJJfkkF+cUV5BWWoNObjky/dorqk5KgtsPhwG63y8WrhoeHGR0dJRQKIQgCOTk5co7vKaecgtvt5oEHHpDD2RQDVkFBQeH4QqVSoVKpUKvVGAwGtFotNTU1rF+/Xq5EGg6HGR4eliN6BgcHefjhh/F6vajVamw2m2wMV1VVUV1dzfr167nkkktQqVSEQiEGBgbo7Oykr6+Pvr4+3n77bUKhkKyV3tDQQGNjI9XV1dTV1VFVVaUYKwqzIh6PK3nmCvNCLBbD6/XOKPmm0WjIy8vLej0ag8HAZz7zGf75n/+Zq6++GovFwmc/+1meeuopbrnlFoxGI11dXdx0001zcr5YLMYTTzzB008/zapVq7jzzjtPmBx7LUAqlaKtrS1jGRjJCAP46KOPMs4J6uvrIxgMsnv37ozDW3w+Hz6fj+3bt8+qrH5fXx+xoBNP9zuk4kGSsTBCIoyYSiCmkqjUatRaAxqtAa3Jht5ahEaXg0anR63Ro9YeWjRaA0VVzeh0OkLhKKFwD0P9PUfcDyGZwOuys337dnp7exkeHsbhcDA+Pk4oFEKtVpOXl0dBQQE1NTWcccYZ5Ofnk5eXh8VikWfl29vbefvtt9FoNDgcjhklRqZjaGiIcDh81McpKCicvKhUKhKJhDw7PRVerxeLxTJn55QGMhNfBUEgHA4T8nkY6m7LuO1YJEA8Fl3Q+rhms1mu8AmHnvHj4+P4/X7Gx8ex2+0cOHCA9957j6GhIXw+H1arlbKyMpYvX87y5ctZunQpl156KSqVing8Lj9b3W43vb29tLW18fjjjzM6OirnCS9evJg1a9awcuVKFi1aJFfgXYi5X+l0WlY3mIpgMDjPPZoaqbCTy+UimUxmtS+iKOLz+RgaGmLnzp3T7mcymViyZMlRtX28yRQdLSqVimg0it1un1LaJZVKEQwGKS8vn9V50uk0w8PDswp1laIGOzs7Z3SOFBUVHbcpEB988AF33XUXgiBMe4+y2WzcfPPNnHnmmVno4WRaW1u56667GBkZITc3l4aGBlasWMEf//hHUqkUl112GZs3b571eUZGRvjpT3/K0NAQX/va11izZk3Wjf65RDs+Po7P5+PWW2/N+GYjiiLDw8MYDAZeeeWVjB9w0WiUvr4+vva1r2XshYzFYvT09PDaa69ldPxEcnJyMITTmNRq0AN6NWD88yIhAOMQH4f45OOTqRR2t5sXnW9lfE3S6TQup5MPnVaGhobkdvR6/SS5H6/Xi9frZffu3VO2EwwGGRoamtVMUCQSobe3N+PjFRQUTj7y8/Opqqri9ddfn3Yfu93OmjVrjrrtdDpNKpUiEongdDoJBoN0dHTQ39/P2NgYIyMjjI6O4vf78fl8qFQq9Ho9Lz/4cTkKURSJx+PEYjEikQjRaJR4PC4btzqdDqPRiMlkora29rgq+CQZpcXFxfJ7n/nMZ+T1cDhMW1sb+/fvZ/fu3dx///14PB4EQaC8vJy6ujpWrlxJbW0txcXFbN68mc997nNYLBYEQaC3t5ddu3bR3t7O7373O+69915isRhlZWXU1tayYsUKmpubyc/Pl6XqTCZTNi6FzNq1a9m3bx+RSGTK7bFYbJ57ND179uzhN7/5DSUlJQtiAuH3v/89v//976fcJooiHo+HP/3pTzQ1NR1xm8ejTNHRIMk1tre3T7ldqrmyePHiWZ3H6XRyxx13UFVVNet71D//8z9Puy2RSGAwGNi9e/dx6Z0/44wzuOKKK7j77rv5whe+wIUXXvixfbRa7awnIeYKlUpFbW0ttbW18nsbN25k48aNc9K+3+/nlVde4d577+WCCy7gxz/+cdbv0ccCrSAImEwmfv7znx/1zJxEIpFg69atVFZWctVVV2X8Q+vu7ua2227jN7/5TcYG+ODgILfccgsffPBBRsdPZMOGDXz1q1/N+IMfHh7mhz/8IQ899FDGYV+pVIp/+7d/Y8mSJVx99dUZPxB27tzJXXfdxf/+7/9mdDxAZ2cnX/jCFzI+XkFB4eRj0aJF/PrXvyYajU67zwMPPEBXV9eM7aTTaUKhEN3d3TgcDlwuFy6XC6fTid/vx+FwMDQ0xKOPPkpubi6FhYUUFxezbt06ORrGZrNhMpnQaDQEAgFZ89bpdDI6OkogECAYDMoyaxaLhdLSUlnL3Gq1YrPZKC8vp7q6eq4vVdYwm82cfvrpcoHCWCzG2NgY/f392O12hoaG+NOf/sQf/vAHIpEIGo1GDnGur69n0aJFbN68mSuuuAJBEOR0m/7+foaHh9m1axfPPvss0WgUvV5PVVUVdXV1tLS00NjYyKJFi+bU038kbN26lUAgMO32/v5+rrvuunns0fSIosjq1av5m7/5mwVvYMRiMbZu3XpUXmnJmz2XOYkLjZaWFp588slpPbDJZJKtW7ficrlmdZ5UKkVNTQ233nrrMfXQjY6O8uijjyJMkYJ3PKDRaPjc5z7HRx99RFlZWcb2z4lAT08Pv/jFL3C73Xzve9/j9NNPP+5zd6dDC4dmEEpLSzN+iMfjcWw2myylk6nRGwqFMBqNVFZWZhw/LknrzAVVVVVs3rw5Y6P34MGDWK1WVq5cmXGfkskkpaWlskRFpkbv8PAwRqNxVgM1v99/wv4QFBQUjg1qtXpGLb90Oo3FYpELKqVSKeLxOC6XC7vdzujoKB6Phz179tDd3Y3FYkGlUpGfn09xcTHl5eWsWrUKnU7HU089xec//3lqamrkInxS5eK+vj527dol10GIRCKIooher6eoqIjKykpOPfVUqqurKSsrk3VspUWj0ZywXqjDMRqN1NXVUVdXBxwyuiKRiBz26HK56OjoYM+ePTz99NPY7XZUKhV5eXksWbKElpYWli1bxqZNm9BqtQiCIOu+2+129u7dy/79+3njjTfw+/3odDo5x3jt2rW0tLTIuXSSpvBcY7PZZjSyZpqkyQYmk4mioqIFH2oYjUaPepwQj8flwpknKmq1mrKysmm3S9rEszV64VBUSmFh4TGtgh2LxY77+6HFYuGMM86goqIi213JCpFIhFdeeYVf/epXnHfeedx2220LxrN9rFAsGAUFBQWFeSEcDjM+Po7b7ZYleMbGxtixYwcdHR0MDAwQi8VIJBKYzWYsFoscHj0yMsI555zDhRdeSElJiVx52Ofz4ff76e7uJhgM8qc//YnXX38dl8tFIBAgGo2i0WjIz8+noKCAyspK1q1bR1lZGRUVFRQWFp7Qg+25QK1WY7FYZG9sfX09p512mrw9Go3S2dnJwYMH2b9/P++88w7btm0jEolgtVopKSmhubmZ2tpaKisrOeecc7jyyivJy8vD4/Gwf/9+9u/fT1dXF//5n/+Jx+NBr9dTXFwse4QrKyvlyfnc3NzjfsCtcIh0Ok08HldkihTmHZVKxZVXXpntbmSF/fv38x//8R8MDw9zxx13TLqfn8goRq+CgoKCwpwiCAJ2u53BwUF6enoYGhqir69PDkUOhULk5OSQm5tLSUkJyWSSiooKLr74YsxmM1arVTZ6jUYjKpWKvr4+DAYDAwMDvPnmm4yOjjI2NobP5yMUCpFOp3G73QwNDbFkyRJaW1tlXXObzUZubi5ms1kxcI8BJpOJlStXsnLlSuBQhJLH48Fut+N2uxkZGZHzhR0OB4lEgvz8fOrq6li8eDHLli3j0ksvpbS0FL/fj9PpxOl0MjIyQk9PD3/84x/lnO2CggKqq6tZsmQJK1asoLW1dUFoaSpkRiqVUmSKFBTmiWQyyXPPPceDDz7Ixo0b+frXv05lZWW2uzVvKEavgoKCgsIRkUwmiUajcqhrNBrF4/EwMDBAb28vg4ODOBwOxsbG5KJRBQUFVFVV0djYyAUXXEBdXR319fXo9Xo0Gg0ajYa7776b999/n4aGBuLxOOFwmIGBAUZHR7Hb7Xg8Hrq6uuju7qawsJDS0lLKy8tZt24d9fX1VFRUMD4+zv3338/f/d3f0djYOEljXGF+0el0sq4w/F/BsVQqRTKZxO/3c+DAAXbt2sXevXt56qmnCIfDGI1GGhsbaWpqYunSpSxZsoTVq1fLBcQEQWDPnj20t7dPOs5ms9HQ0CBXjy4rK5PDl81msyLPt4BRZIoUFI49oijS19fH1q1bsdvt3HHHHSxfvvykS1k8uf5bBQUFBYVPRBRFxsfHcblcjI6O4nK5ZF1wh8OB0+kkmUwiiiI6nY6ioiJKS0tZunQp5557rqwFXlpaSl5e3qS2g8GgbMxKFZZ37NhBV1cXg4ODiKJIOp3GZrPJXr2VK1ciiiIXXXQRW7ZswWAwfMygTSaTaDQa2ZhWWDhInjydTofJZJJ1gs8//3zgkOHT39/PwMAABw8eZHBwkMcee4zx8XHUajVWq5W6ujpqampoaGjgvPPO4wtf+AJGo1E+pre3l/7+ft5++21isZhcsbq+vp6Ghga58ulcVLVVmBtOdJkiBYWFgCAIvPTSS/zmN7/hrLPO4tvf/jalpaXZ7lZWUIxeBQUFhZOIdDo9aYlEIvT399Pb20t3dzf9/f309fXh8/mIRCJotVpKSkooKSmhtraWjRs3UldXJ3vRJoYiq9XqSW37fD7efPNNent75ZxPKUxVrVZTUlJCVVUVADU1NVx++eUYjUbMZrNcQVkKe/zggw/Iz89f8IV8YMI1FkXSaXH27aTTJ7TX2mAwyLrCF1xwAalUikAgIGsLOxwO2tvbaWtr4+mnn8bn85GTk0NNTQ1Llixh5cqVXH755eTn5+P3++Vj+/r6aG9v56mnnmJ4eBiA0tJSWlpaWLVqFatXr6a2thadTodKpfqYMTzx+k/FQtJqXih9Ofz+Mh2S/vNUE1gnM5/0fZuLz3mu2jjSz1khM+bic/L7/XR2dhKJRPjHf/xHNm3adFKnEihGr4KCgsIJRjqdJhwOEwwG8fv9BINBAoEAo6OjDAwMMDAwgNPpxOv1EgwGsVgs5ObmUlNTQ11dHWeffTaNjY3U1tZOWXlZFEUCgYCcfxkIBHA6nXR1dclhyZI3OC8vj6KiIhobG7nkkktobGykpaWF4uJieUB099138/DDD0+rYQmH9CePF0RRZKSvg1e2/RqDKXMpHpd9hA8//BBRFLFarVgsFnmi4VhUNF4oaLVaCgoKJmnRX3zxxfJ6OByms7OT3bt309HRwX333YfL5UKj0VBZWUlNTQ2tra00NDSwbNkyioqKyM3Nxe12s3PnTtra2ti2bRu//OUvSaVSVFRUUFtbS2trK4sWLaKwsFD+3ezatWva8NuRkZEFkQ+XTqdpa2vjwIEDWTd+k8kkbreb//iP/5g211qlUrFkyRLWr18/Y2X3kwlp0qWjo4NoNDqlsZlzslQAACAASURBVBiPx/H5fLM6TzqdZmBggA8++ABRzHxCzufz4XK52Lp167RGlEajYcOGDaxfvz7j85yMBAIBfD4fv/rVr2bVjtfr5fnnn0cURW688UY5suZkRjF6FRQUFI5jEokEY2NjjI6OMjg4iN1uZ3h4GIfDgcfjIR6Po9PpMBgMFBYWUl5ezsqVKykuLqaoqEjWoC0sLJxy8CIIAg6Hg8HBQQYGBhgaGmJoaIixsTG8Xi9qtRq9Xo/NZqOmpoZly5Zx3nnnUVpaKksKfVK13Q0bNmC324nFYtPuczyFpFZVVbFx/Wrs9v2M/NlbKU0wiKKI0WgkNzeXgoIC8vLy5GJbh7Np43qSySQ7duwgHo+TTCZJJpNYrVYKCwsRRRG3282BAwfIyckhLy/vpPCsmM1mVq9ezerVq4FDvwEpZH5gYICRkRF27tzJ888/L3+nysrKqKmpobm5mYsuuogbb7yRZDLJyMiI/L3eu3cvL7zwArFYjEgkIlcFr6mpobKykoqKigWZfyqKItu3bycUCsmRE9lk48aNjIyMMDIyMuX2UCjE9u3bWb58OcXFxfPcu4WJRqNh8+bNhMPhabVvI5EIbW1tsz7Xzp07ee+992htbc34fqHT6TjzzDNnnKiUflennXaaknJyhITDYZ555hkaGhoIhUKzakutVvPpT3+aoqIiucjgyY5i9CooKCgsQKSiP4lEQjZ2XC6XHIo8MDDA8PAwY2NjJJNJVCoVJSUllJeXU1VVxapVq6ivr6e4uBiDwYDRaPyY7qkgCCQSCRKJBD6fj2g0it1u58CBA/T09DA4OEhvby/JZBJADkeur6/nnHPOob6+HpvNJrdtMpnQ6/VHPZBau3Yty5Ytm9FLdfXVV2d+MeeZuro6brrpJuLxuPw5JpNJYrEYDoeDkZERnE4nY2NjOBwOxsfHMRqNFBUVUVZWJudDV1dXk5OTgyiK8pJMJuVjOzo6GBwc5Hvf+x4ajQaLxSJ766uqqmhubp6kOazT6dDr9Sdc8RK9Xi/rCq9fvx5RFCcVXPN6vezbt48DBw7wu9/9DofDQTqdpri4mKamJpYtW8aGDRvYsmULgiAgCAKBQID29nYOHDhAe3s7IyMjjI2NUV9fz+LFi1m5ciV6vZ533nlH1pnNppZzOp3m4osvZvPmzVk5/9EwNDTE/ffff0JHKxwtarWaTZs2ccYZZ0x7H5T0sWeLKIqsW7eOa6+99pjeC1588UW8Xm/Wow+OJ/7whz8wMDDAz372M2pqambVlpQWo1arj4u0oPngxHryKSgoKBxnCILA+Pi4XDhK0rGVPCVut5twOEwkEpH1ZouLi6mpqWHNmjVycZ7KysoZB5GxWAyXy4XH48HlcuF0OrHb7fT29uJ2uwmFQkQiEUwmk6xhu379eq655hpqamqoqak5Zl4ulUqFyWSacZ/jydMrDTKmGmgsWrRo0t+iKMrFvSRj1m63s2/fPv4/e/cd3mZ5Ln78qz2saXnIkrcd2xk0k50QNqVAaVMKPb+20EF7IIyWthAIqwmUlAIhZTRh5wAtbSklEEIoNNCGpKE0O2Q6HvGWZGvv+fsjR++JySCYJM54Ptely7b06tUjWXr13s+471AohEKhkNY3FxQUSCP0+VHHaDTKvHnzsFgsdHV10d7eTnt7O//5z3948803icVi6HQ6jEYjer2esrIynE4nhYWFFBcXY7PZsNlsx9UosVwup6CgQEqQVFVVxfjx46XbE4kELS0tbNu2jZ07d7J8+XL+9Kc/kUqlMJvN2Gw26uvrqa6u5vLLL2f69Omo1Wq6u7v55JNPaG5u5s9//jMbNmygv7+faDQq1YE2m80YjUZMJtMRHxVWq9Wf+Tk6GuTX8R5vnS9fVP6zvj86ne6QHQfzSeUO5//gRF47OhSbNm1i/vz5PPDAAzQ1NQ13c45L4ogjCIJwhASDQTo7O9m1a5dUv7ajowOv1yvVrjWZTFgsFmm01m63S1Ng8yfVBQUFBwxQotGolM12x44dtLa2SjVTM5kMFosFm82G3W5nwoQJ2O12rFYrNptNOnkXJyxHhlwux2w2YzabGTlyJLC7hz6VShEOhwmFQtKa7D2TOkUiEXp7e2lra+Omm26ioqJCGoWfNGkSX//617HZbITDYfx+PwMDA/j9frq7u9m2bRs+nw+v10sgEBiUVKyuro6amhqqqqqorq4+bgLhPWk0GkaNGsWoUaOA3bMq8p1BLpcLl8vF1q1befPNN+nv7ycUCmGz2aisrORLX/oSl1xyCRUVFbz44ou8/vrrNDU14fP5aG9vJxwOE4/H0Wq1WK1WvF4vfX192O32z/zcnkiOpU4sQTjcXC4Xs2bN4oc//KFYA30YiaBXEAThC8hmsyQSCRKJBPF4XFpP2NnZSXt7u7TOtq+vj3A4jFqtxmw2U1VVRUVFBRMnTqS2tpbKykr0er1UuzZ/2fMkOZfLEY/HGRgYkEZmfT4fLS0tUpDb19eHz+dDpVJhNpupr6+nvr6eL3/5yzQ1NVFUVIRKpZL2r1QqxYn4UWbPGsd7JnPK5XJks1kymQzZbJatW7fy2muvceuttxKJROjo6GDLli28/fbbDAwMkE6nKSwsxG63S2tSGxoaOPXUU6Upz2q1mmg0SkdHB+3t7WzZsoUlS5YwMDBANpuluLgYh8OB0+mkoqKCuro6adS4oKAAvV6PXq8/pt9DSqWS0tJSSktLGTNmDIBUVzifSXr79u188skn/Oc//+HVV18lFArhcrmIx+PS+vV8/WmZTEYgEMDj8QCwYsUKVq1ahU6no7i4mJKSEpxOJ0ajEYVCIf1fTyTH8vtFEA6lSCTCvHnzKC8v59vf/rZY/3wYiaBXEAThIMViMfx+P16vl/7+frxeLwMDA/T39+Pz+di8eTN+v59//OMfWCwWSkpKcDgcTJgwgdLSUpxOpzRy+1knfflaud3d3VLQ3NXVJdXKTSaTZLNZqaSQ3W5nypQpOBwOysvLKS8vP66mrAq7A4V8ZwWAVqtFrVbT0NCwVxbhWCxGT08PfX19Uj3ktrY2PvroIyk7bH5fVqsVp9NJUVERF154IaWlpRQUFJBMJnG73fT09EidOX/729/IZDLI5XJpGnFZWRllZWXS+91ut1NcXHxMr9lUKpXS1E+DwYDD4ZDWy6ZSKTo6Onj88cdZsmSJlGAokUhINYnNZjMWiwWVSsVZZ52F0+kkFApJx42WlhbS6TQAPp+P999/n+rqakpKSqTp5sfzaKg4LgnCbosWLWLbtm3MnTtXTPk/zMSrKwjCCe3T9QbzF4/HI53s54NNt9tNPB5HJpNJU4GLiooYOXIkpaWlxONxrrjiCr7xjW9I9Wv39yWWzWalx8pkMvT397NlyxZaWlpobm5m586dUvZlq9WK3W6noqKCcePGUVtbi8VikdYOGgwGdDqdOJEUJDqdjrq6Ourq6qTrstks0WiUcDgslbTKrydua2tj3bp1UpCcTCYxm804nU5qa2upq6vjjDPOoKSkBJlMRjQaJRQK4fV6aW9vp7m5mZUrV9LT00MoFEKv11NVVUVNTY0026Cmpgaj0YhMJpMSrByL71mVSiVNA3c4HJx33nkkk0ni8biUQMvlcuF2u4lGo/z973+XMm47HA7q6+spLCxELpcTDod5+eWXSSQSrFu3joGBAWkJQnl5ORUVFdKoe36EP/+aHYuvnSAI/2fLli288MIL3HPPPVRXVw93c457Sth90uf3++nv7x/STpLJpPRF2t/fP+TeSZ/PRyqVkk70vsg+DoX8NMKhZj3z+XzSWqGh9nin02lisZj02g71S87v90v184bK7/fvN5X+kZZ/HaLR6H7blEgkRNZAYZBUKkUkEpFO+KPRqFS/1uVySWscA4EACoUCvV4vjdBOnjxZShhlNpv3OQXp7bffxm63DyobkkqlCAQCBINBaf+9vb00NzdLyYvyn8uioiKcTicjRozgvPPOk4INsb52d8AWiUT2W6cyGAyecFNEPy+5XI7BYMBg+OzawaFQiObmZnbt2kVbWxsffPABPT09BAIB1Go1VquVkpISCgsLqays5Nxzz8VoNGI2m1Eqlfh8Pnp7e2lvb+fDDz/k5ZdfJhgMUlBQQGFhIUVFRZSVlVFVVUVJSYm0rtlisWAymY6ZUeJ8Vm6ZTIZGo0Gj0WCxWHA4HORyOdxuN5dccglGo1HqRGttbZWSlOX/F3a7naqqKgoKCqTzoL6+PtasWUNHRwfxeJxgMEhZWRkVFRXY7XZMJpPU6XUwotEoyWTycL4cByUUCpFKpfD5fPudypl/r4pj397yy2T2JRKJHJLHiMfjBywhdzCi0SiJRAKv17vf/7NSqcRgMJyQU3o7Ojq46667uPrqqznrrLOGuzknBKVCoSCRSDB79mzMZvOQdpLJZNi2bRs6nY5//etfQw7MQqEQO3bs4Oabbx7yEH8kEmHnzp1Duu+nrVy5khtuuGHIH8ZwOMyuXbuYPn36kDsCstksmzdvxmAwsGLFiiHtA3YXqd6+fTs//vGPh7yPYDDIwMDAkO9/KMnlctLpNEuWLNnvaxsKhYb8nhaOfX6/H4/Hg8fjoa+vj/7+fik7ciKRQK1WS/Vri4uLsdvtjBo1CovFgsViobCwUDr5PthjWjabpbe3l+XLl9PW1kZXV5dUMzeZTEonxYWFhVRVVTF58mTsdjslJSVSG8T0pn3LZrO89tprLF++fJ+3x2Kxo+KE/nhhNBoH1cKF/0v41N/fz8DAgFQLur29ndWrV0vr2hOJBBqNhtLSUux2O2eccQbTpk2TyiaFQiHpc7lu3TrpM5lMJkmn01itVikgdjqdVFZWUl5ejtFoHMZXZG8KhYKdO3fi9/v3e4zweDyoVCppyQHsPmfKd67lX79du3bR2tpKIpEgm81isVgoLi5m/PjxFBUV4fF4mDx5Mn6/n9bWVlavXk0ymSSXy1FUVERzczPl5eXU19dTVla213Erk8nw2GOPfaGO70MlGo3S3t7Oz3/+8/1muJbL5VxzzTVccsklYkT7fykUCuRyOX/84x/3G/R+0UA1v4+//vWvrF279gvtJz9j5Lrrrtvv/1CtVnPnnXdKa+lPFOl0mvnz5+N0OvnGN74x3M05YSjNZjMmk4l7772XkpKSIe0klUrx+OOP43A4uOqqq4Z8gGptbeXee+/l3nvvlUoNfF5dXV3MmDHjkARnEydO5J577hly2YFdu3bxs5/9jPvuu2/IvZXpdJq5c+dSXV3NlVdeOaR9AKxfv55HH32UBx54YMj76Ojo4Gc/+9mQ738o1dbW8pe//OWAB/h//OMfPP3000ewVcLhls1mB9U9TafTxONxent7pUtfX580rRDAZDJRWlpKUVERY8eOxel0YrVa0el0Ut3SfO3S/XWg5Kcg71kzNxqN0tfXR0dHBz09PfT29rJmzRpaWlqoqqqSTtjPOOMMRowYQXFxMTqdTiplo9Vqj+s1e4fak08+STgc3u/tnZ2d3H///UewRSeePRM+7Sk/2rln0DswMMDOnTvp7u7m3//+N52dnXi9XmQyGYWFhVRUVFBZWcmYMWOk2RPZbJZ4PE5fXx89PT1s376d999/n56eHjKZDGazmZqaGioqKqiqqqK+vl6qQ71nJ9aRGjX6zne+w4UXXnjAGUU/+MEP9rpOoVBIScqcTif/+Mc/uPTSS7FYLCQSCaLRKF1dXfT09LBhwwba2tqIx+PI5XIqKiqoqKjg1FNPRafTEY/H6ejoYNu2baxatYo1a9agVCqpqqqisrJSKr0kl8vZsGED119//Reu/3m4ZbNZlixZwrp167j44otPyFHAfbFarbz88svSd9u+tLW1ceutt36hx4nH46xZs4Yvf/nLe5VYO5RisRjPP/887e3tJ1TQm39/b9myhUcffXTI8Y7w+SnlcjlKpZL6+nocDseQdpJIJLDZbJSWltLY2DjkE7lcLoder6ehoWHIPbpqtRq9Xj+k+36axWKhoaFhyHXvFAoFWq2WhoaGIQfOqVRKyr7Z2Ng45A4Fn8+HXq//QrW/lErlUVPgWqVSUVVVdcBtmpubRQ/xMSw/9TgQCODz+aTpwS6Xi0AgQDQaJR6Pk0qlMBgMmM1mSktLmThxopRQp6ioSKoJebByuRzRaFSaipwv7eJ2u3G5XESjUWKxGIlEAoPBINVMHTVqFLFYjGuvvZbvfve74r13iH06UdOn7Zl4SDiyVCoVKpVq0Pd2XV0dp5xyyqDtkskknZ2d0gyI9vZ2/v3vf+PxeEgkElIt4nyW46amJk4//XSsVqsUDAYCAfr6+vjnP//Jq6++SjKZHJRFOj/112azUVxcTGFhISUlJQc9BfjzyM8KOZCDOR/JZ+s2Go3Sa5gPTHO5HB999BHr169nypQpuN1uWltb+fjjj8nlcuh0OkwmE9lslpNPPplJkyYRiUSk8laLFy8mFAqhUqnweDx8/PHH0jTp/Gt0tJ10ZzIZCgsLpbwHwm75To8DOZSvV1lZ2WFdZxqJRDAajSfc/3j16tXMmzeP3/zmN9TW1g53c04o4gxBEIRhlclkpBHa/Jo3l8uFx+MhGo2iUqmkdWtWq5Xy8nLGjh2LyWTCaDRiMBgwmUzodLohdbgFAgFpVKWjo4Ouri4GBgaIRqPo9XqMRqOUifm0007DbDZL1+VLt+QD3OXLl3/uIFsQThRqtXqv5Fq5XE6qJRwIBKTOpr6+PlpbW6Vs6fl8HVarFYfDQW1tLaeccopU1xp2L2nYtWsXn3zyiXQfv9+PRqOhpKSEqqoqRowYQVVVFbW1tdjt9mPisyqTyTCbzZx66qlotVqy2ayUJyC/jKO5uZnNmzfT1tZGJBLBbDbjcDgYM2YMNpsNrVZLW1sbCoWCTZs28cEHHxAMBqWOgpqaGhoaGqipqRH1hAXhMOnr62Pu3LlcccUVnHzyycPdnBOOCHoFQTgs0qkkqUSMZDJOMh4lmYgTC/kZcHUSGOjF3dNG6+Y1zJq1HYPBgF6vx26343A4mDhxojRio9frkcvl0nqmfMbXzzopy2QyUjKO/Ois1+uV1tj29vYOGmEqLi6msrKSU045RVpHqFar9/nYgiAcGjKZTBrh3HMUK7+kYM9LNBqlublZqim8fPlyenp6pMQ+paWllJWVUVRUJE2b1mg0+P1++vr6cLlcrFq1ij//+c/SOtySkhLpWJPPNp2vQbznCPLRRC6XY7VasVqtVFZWkslkWL9+PWeffTZTp04lkUjQ0tJCW1sbbW1trFy5UupY8Hg81NXVcdppp2EwGBgYGKCrq4vt27fz97//nVgsNqiOeF1dHVarlYKCAgwGg1iWIQhDlEqleOyxxygvL+e73/3ucDfnhCSCXkEQvpBcLksk6CMcGCDk6yfocxHyDxD0uQgHBsikkmSzOXJkUShUGC1FGMw2nDWjCPd38IPvfJNJkyZhNBqHvHYrnU7j9/ulTMj5zKcDAwPEYjEymQy5XA61Wk1RURFFRUU0NTVRVFRESUkJRUVFIkuoIBxFZDLZXlPWjUbjXuuJs9ks/f39Uv3qfKfW6tWreffdd8nlclInmV6vl5Yi5EdM8wF1MBjkX//6F4sXL5buk8+uXF5eLtXCdjgclJaWUlJSclQEf/nnlq8PnA/SJ02axKRJk4Dd08p7enr4+c9/Tk1NDalUivfff59EIoFcLketVlNcXMzUqVORyWRkMhnS6TQ9PT2sX7+edDotdU44nU7KysooLy+nrKzsuK8nLAiHQi6X44033mDDhg088cQTmEym4W7SCUkEvYIg7NPuGrL/V0uWXJZkIo63rwOvp4v+3g68rg587m6i4QAyGRSYCjFaSjCYCyl21tI4bgo6gxmNzoBao0Oj1aPWFaBQKPH399DXspby8or9rovLr/XJr+3KZrNS6a09syN3dHQQjUaloNZms0nZmG02m7ROMP9ToVCIEVtBOA7I5XJKSkr2SsSZTCalesThcJhwOIzX66WlpUUqUdbX14fX65WOG2VlZYwcOVLKUWIymaT7fPLJJ7z//vv09fURi8UwmUxUVVVRXV1NY2MjtbW1VFdXo9VqpRrER0swqFarKSsrw2AwcMEFF1BZWTnotfH5fNLoeb6zMD8lvLy8HLvdTnFxMeFwmLa2Nv71r3/R29tLKpWiuLiYESNGUFdXx8iRI6X6w3s+f3GsFU50a9eu5YknnuDBBx+kpqZmuJtzwhJBryCc4HK5HMl4hFg0TCIaJhYJEo+GCHrdeD1dRIM+wgEv0YifZCyGVm/AYLFhs1dR3TSRCVMup8hZTYHJ9oWzbCYSCUKhEOFwmGAwSDAYxO12093djd/vlxJaKZVKiouLcTqdnHTSSVx66aWUl5djMpmOmhNNQRCGj1qtljIkH0gmk5ECvvb2dnbu3MnmzZtxu92kUinMZjNFRUUUFhYyfvx4KRPynqWbXn/9dQYGBgiHwxiNRmw2G0VFRdjtdrq7u7FYLGg0GnQ6nZTBfTiXSsjlcqnGb96eScdSqZQ0NXrnzp2sXLkSv9+PXq+XOgROPfVU5HI5brcbt9vNe++9xx//+EeUSiWFhYVSySm73U5hYaFUh1kkmxNONL29vTz88MNcffXVeyX3E44scfQRjlsymYxkMonb7d7vNrFY7Ai26OiQSiVp37aWWDRM2D9AOOQl7O8nEYsCOWD3iZhaq8NgKqTAaKW0oh690UqByYrBbEOp0rDn6VrifwPmz8PvdREJBVi/fh2bN39CX18fHo+HcDiMQqFArVaj0WikxDWjR4+WTmKtVisGg0GMIAiD5Kdxnijvi97eXrLZ7HA345imUCioqanZa/QlFApJQW1+2YTH45HWx+bzBaTTaWw2G01NTWi1WqkWt0ajIZVKkUwmWbt2Ldu3b5cer6CgQErCp9VqyWQypFIpaVr1cFOpVDQ0NNDQ0MBFF10kLR/JJxgcGBhg69athEIhKYO+yWRi5MiRaDQaMpkMsViMFStWEIvFyOVyUkdlvuTVyJEjRebawySRSNDb27vPahu5XI5AIPCZnUHCoZHNZpk/fz5Wq5Wvf/3rR8Xn+0Qmgl7huGUymUin0yxatGi/27hcrsNah+5os3tUN0b7tn+y65PlZNIpMpk0uWwWmVyOQqFEoVShUKhIKZXEvZ30H6aDdDabIREN8fHHH1NTU0NJSQnjxo3D4XCg0+mk+rn5+pviy0I4kPz75LnnnjthRpNSqRTjx48X69EPg3xyrT2D4Ww2SyKRGHSJRqO0tbVJpZhaWlro6+sjGo1K64a9Xi89PT0kk0lgd3mt/MivVqvF5XLx1FNPUVhYOKjMUr6cUDQaJZVKkUgkUKlUR3yUWKlUSrkQYPf3SP75x+NxQqEQHR0dtLa20tHRwYaNG9EWWMjldndEJZNJUuk0bGtFpVSiUMhpqK3gjjvukPYpHBr58lPPPvvsft8jbrebSy655Eg264SUyWR49dVXWb9+PQsWLMBqtQ53k054J8aZgXBCmjJlClu2bDngNj/5yU8IBoNHqEXDT6VS8fDDDw93MwThkKuoqGDZsmXD3QzhOCaXy6Upyntqamraa9v+/n46Ozvp6emhra2N7u5u+vr68Pv90mhvPuh98cUXaWhoQKVSEY/HpeA5nU6Ty+UIBoMkEglefvll7Ha7VB84P0X5SNc6lclkaLVatFqtVB+9vr6ec889l2g0yg03/YSLf/QQ9sp9dyi7u1tZ9vxtoiPzMKioqGDHjh0H3OaRRx6hubn5CLXoxLV69WpefPFF7rnnHhwOx3A3R0AEvYIgCIIgCIdUfmR0/Pjx0nXJZBK/308wGJRqCLvdbhYuXEgsFiMajRKJRIjFYsjlclQqFTqdDo1GA4DVaiWdTtPa2kogECASiRCNRtFoNLS3t6NWq8lms5SVlVFRUSFyHAjCMInFYjz00EOcf/75TJw4cbibI/wvEfQKgiAIgiAcZmq1eq9M07FYjLvuuotLLrmE4uJiqT6x1+vF4/HQ39/Ppk2bpDJMuVwOvV6P1WodVDLI6/XidrtZunQpXq+XXC6HTqfDZrNRUlKC2WwmGAyyc+dOstkser0erVYrLSURo66CcGjEYjHuv/9+KioquO6668Tyk6OICHoFQRAEQRCGSb4msVqtlq7T6/WUl5eTy+UwGo3Y7Xa+853vEIvF8Pl8DAwM4PP58Pl8eL1eKXN0QUGBlOQvm82SzWbp6+uTMuAvXrwYo9EI7J6ubTQaKS4uxmazUVxcLGWqtlqtIhAWhM8pk8mwaNEi1q9fzwsvvIBerx/uJgl7EEGvIAiCcFzJ15be39TOTCZDJpOR6qkeqNTWZ2XUza+l/LwBwuHar3D80ul0Ujb7vGw2SyQS4YknnuCUU05hwoQJRKNRent7cblcUmklv99PMpmkt7eXRCKBzWbDaDSi0+kIBoP09vbS39+Pz+cjk8lII8nl5eVUVFTgcDgoKytDqVRKibTyP8V7VBB227p1K8899xy/+MUv9qodLgw/EfQKgiAIx4VwOMzGjRvZvHkzAwMDnHLKKZx99tlS8JtIJFi7di3Lli2T6o6aTCYmTpzIlClT9sr6HIlEePvtt5k8eTJlZWV7PV4ikeCtt96iqqqKSZMmHXQ7g8Egb7zxBpdffvmgWql5yWSSpUuXUlRUxJlnnvk5XwXhRCKXy9Hr9ajVaoqKimhoaABg3Lhxg7ZLJBL84Ac/4Pvf/z5yuZzOzk76+vpoaWkhEomgVCqxWCyMGDECpVJJKpUiFAqxatUqli5dSiqVQqPRYLPZMJvNWK1WbDYbDocDs9lMQUEBJpMJhUIhymgJJySfz8c999zDFVdcwfnnnz/czRH2QQS9giAIwjEvkUiwcOFC3njjDU477TQCgQC/+MUveOyxx5g8eTLxeJyXXnqJ9evX8+Uvf5na2lr0ej0ffvghL730sftqAAAAIABJREFUEt3d3Xz729+WRq3y09QeeeQRnnrqqX0GvatWreKmm27i9ttvZ+LEiQc14pXJZHjzzTeZOXMmZ5111j6D3o8++oif//znXHPNNSLoFQ4ZpVJJfX091dXV0nWpVIpAIEAgEJCSawUCAfr6+qRp10ajkXA4TDQaxe/3EwqF8Hg8aDQaNm3aJNXmTSQS5HI5ujo7+OjdP1JeNxprsQNLsROTtQSlSqwdFo5P8Xicxx9/nNLSUv7f//t/J0zZvGONMpvNkkwmeeedd4ZcQyqVSrFjxw68Xi+LFi0a8kGtq6uL/v5+Fi9evFdJgIOVn85zKLS3t/Pmm28OWmfzefT29uLz+XjzzTeH/AHIZDI0NzcTiUR4/fXXh/zabt++HY/Hw+uvvz6k+wP09fXh8/mGfP+jkUwmo6Ojg9WrV+/z9kAgQCgUOqLlIARB+PxaWlpYuHAhv/rVrzjnnHPIZDLMnDmTDz/8kMmTJ7Ny5Uree+89Zs2aRSgUYuHChSQSCS644AKmTp3Kq6++SlVVFZMnT+bpp5/m+eefx+127zcJSU9PD48++uigACKRSBAMBjGZTGg0GqncTDabxWq1snDhQp5++ml6e3vp6+vb535dLhcLFy5Ep9OJAOEEIJPJiEQibNq0ic7Ozr1uz+VytLS0HLb3gkqlGlSDN/+YqVRKuiSTSVKpFAMDA/T09OB2u6Xp0/n6w/mkWTqdDhnQ3fYJHc3riQS8JBNRlCo1psJSTNZSVBoN/O8SBOHIk8lktLW18Y9//GOfSzsikQgej+dz7TOfgC2TyZBOp0mn04TDYVpaWshkMnttn06n8fl8tLW1Dfl5HC2y2SyLFi1ixYoVLFy4cJ8dmcLRQZlPe7906VJsNtuQdpLJZGhtbcXtdgNDX4OUT8qwbNmyIWc7y5cCOBS6u7t57733hhyw5gOmd99994Brxg4km83S3t6O3+8nm80O+bV1uVx4vV7+9re/Den+gJQd8nhy1llnkUgkiMVi+7w9kUjs84AtCMLRZd26dTidTqqrq/nkk09QKBTcf//9FBQUAPDSSy9x2WWXEY1GmTFjBt/85jcJhUL87Gc/Y/r06Zx77rn8/e9/54wzzmDKlCk4nU48Hg9z587d67GCwSAPPfQQ5557Ltu3b5eu7+npYcaMGVx44YV8//vfx+PxcNttt3H66adz/fXXc/rpp1NcXExLS8s+95tKpViwYAEjR44c8neGcGxRKpVcccUVZLPZ/X4PpVKpIXe+D4VMJkOtVu/1mE6nky996UuDrovH43g8HjwejxQIG43GPb43K/fxCDnKx5whstoOkwkTJtDc3LzfgDMejxONRkkmk1KN6EgkIpXTikajJBIJkskk6XSaTCZDMpkkFosRCAQYGBjA7/fT1dVFT5+bAqMZtbYAtUaLSq1DrdWj1miRyZV09npJJpNH+BU4tDZv3szChQu5/fbbRT3eo5wyk8mg0Wi46667GDt27JB2kkgk+OUvf0llZSX//d//PeS6cFu3buW6665j3rx5UnbBz6u1tZVrrrlmv73on8eZZ57J448/PuRR5+bmZnbs2METTzwh1dn7vFKpFHfccQcNDQ386Ec/GnLQu2rVKmbOnMmCBQuGdH+ATz75hP/85z9Dvv/RaNq0aUybNm2/t7e3tzN9+vQj2CJBEIaiubmZeDzO7NmzpQy3jY2NPPjggxQUFNDW1sZJJ53Eb37zG66++mquvvpqAoEAK1euZPTo0QQCAdrb28lkMowaNYpRo0axa9cu5s2bN+hxcrkc77zzDqFQiGnTpjFnzhzpturqar7+9a/zyCOPMGnSJKl8zFe+8hUAGhsbaWxsZN26dTzzzDN7PYe//e1vbN26lfnz53PnnXce3hdMOCoolUqefPLJA27z6quv8j//8z9HqEWfj1arpaKigoqKiuFuivAZkskkgUAAp9PJtddeSyAQwO12MzAwgMfjkWo/h8NhUqmUdO6aSqWIx+PEYjHi8TjJZFJKxKdSqdBoNJhMJkwmE1arlZKSEkaNGkVrayvbu0NM/eoP0eqNKFVqVCoNSo0GlUpDIh7nr8/MPqYHFsLhMI888ggXXHCBWIpyDBCTzgVBEIRjns/nY+3atfziF7/g5ptvpr+/n+9973vce++9zJo1i0wmg8vlYuvWrbzwwgsoFAr8fj8Gg4GmpiYWLFhAeXn5Z46o7dy5k4ULFzJ79mwKCwsH3SaTyZg2bRobNmzgJz/5CZFIhAULFlBVVfWZ7d+wYQMvvPACd99995CXGgmCcGzLLzlMpVKk02lpNHXPaeb56/JB7KfXYw8MDEiBrN/vJxKJEI/HyeVyUmkstVqNRqNBo9GgVCpRqVQolUrpb5lMRllZGXV1dRQWFlJSUkJZWRk2m23Q7Md89u5PX+RyOUuWLMEVb6F21MkolHuP6ufwIxviINnRIBqNMmfOHLRaLTfffLOYuXAMEEGvIAiCcMzTarWMGTOGb33rW+h0OioqKrjiiit49dVXCYVCmM1maTqf1+vFaDQyf/58DAYDiUSCjRs38uCDDx7wMdLpNA8++CB2u51QKMSKFSvo6elBqVSyceNGxo4di0ajYfLkyTz77LNMnDhxr+mg+/P888+j1+txu93885//pLe3l2QyySeffEJjY6M4oRKEY1AmkyEajUrTg6PRKOFwWPo9FAoNui4Wiw26PRwOE4/HCYfDhEIh6ZIfbdXr9RQUFGA2m7FYLFitVsxmMw6Hg8bGRgoKClAqlWi1Wmlbi8WC2WymsLCQoqIiqbZz3sDAAD/60Y/42te+xpgxY4b83I/3nARvvfUWW7ZsYe7cueL4fIwQQa8gCIJwzKupqWHdunWDltcYDAZyuRzZbJZzzjmHlpYWJk+ezLe+9S3sdjsymYyuri4efvhhpk+fTlNT0wEfI51OA7uD5qeeeopMJsPmzZvp7OyktLSUsWPHEg6HefbZZ7ngggvYsGEDr732Gt/61rc+8wQwkUjg9/t5/vnngd3LSXQ6HUuXLqWyslKcVAnCYZSv7X0wl1AoJI2qer1egsEgfr8fn88n5ZXx+XzSKGx+pDa//lUmk5HL5Uin0yQSCeLxOKlUikwmg0qlwmKxYLFYpJJQVquVoqIiCgsLpVJRBoMBhUKBSqWSphjveVGpVKjVapFF+DDp7u7mqaee4qc//elBzeQRjg7i0yAIgiAc88477zx+97vf8de//pX/+q//ore3lz/84Q9MmTIFh8PBd77zHe644w7Gjh3LKaecwrhx46ivr2fJkiVEo1E++OADJk2aRElJyX4fQ6vV8uyzz0p/RyIRbr31VpqamrjppptIp9M8+eSTRCIRHn30UZYtW8bcuXOpra3l1FNPPWD7P51vYfr06TgcDm699dYv9sIIwgkon2gpvwY1H1zmr0skEtIln4DJ7/cTDAYJhULSaGsgEJCC2nyCpnxpJplMhlarlWoUFxQUoNVqUavVyOVyFAoFZrMZrVaLwWDAYrFIo6v5wLaoqEgadbVarUPO/yIcOS6Xi5tvvpkrr7ySyy67bLibI3wOIugVBEEQjnlVVVXMmDGDJ598kr///e94PB4puaJSqcRutzNz5kz+8Ic/sHHjRrq6uojFYnR1dVFbW8uNN954wID3YKxcuZKXX36ZBx98kJqaGr7xjW/w3nvv8etf/5pXX31VjLoIwhDly4Hlp/nmf9/XdaFQiFgsJl1SqRSANLqazzScD2rzU4nzWYTVavWgUVWz2Ux5eTkGgwGdTofBYMBsNmM0GjEajeh0Omnq8J4/89vqdDqRjf04EY/HeeaZZygsLOR73/vecDdH+JzEN7AgfIZcLif1FO9LKpUS9QYFYZip1WquvPJKzj33XGKxGHK5HLPZPCgp1IgRI7jttttoa2ujq6sLgLq6OsrKytDr9Xvt0+l0smTJEoqLi/f5mHq9ntmzZ6NWq5HJZEyYMIElS5ZIZSusVisLFiwgFAoNCnhHjx7N0qVLsdvt+30+s2fPFkGyIMlkMiQSiX1Ok8/XRz1a5EdS80sL9nXJ13HNj6zuub41X7c4k8mwYsUKgsEgsVhMSuSUzWbRaDTodDop8VI++VI2myWVSkmjuOFwmFwuh1qtRqfT7XOq8J5rXAsLC7FYLFJiJ4VCsdfPPX8/nuVHxYdq95TuFPFYZJ+JrOKxCJl0SsoOfbjkp46n0+khP05+DfU777zD6tWrmTNnjhiVPwaJb1RBOAC5XE4mk2HRokX7LaPldruJRqNHuGWCIHyaSqWirKzsgNvodDqpJNFnUSqVlJeX7/d2mUxGUVGR9Hd+5GdPVqt1r2zMarX6gPsFBu1XOLHJ5XK2b9/Oyy+/vM+OkFwux86dO7nooouGoXWDpdNpfv/737Nx40bi8bgUAH/69z1rs+brAmu1WmlNaiwWk2psl5aWolKpUCgUUmCbyWRQq9UYDAZperHFYsFkMmE2mzGbzXtdt2eyJmH/5HI5crmcv/zlL9hstiHvp62tjdYON3995v59ZmlOJeN0t2zklVdeYe3atV+kyQeUzWZZsWIFer1+v7WJP0u+tNP8+fOZPXs2I0eOPMStFI4EEfQKwgEUFxdz77334vF49rvNxo0bWbp06RFslSAIgnCiOOOMM3j00Uf3O6Mom80SDAaPcKv2LZlMsnzFSkacNg1L0YE7oPYnm82wcdW7lGij3HzzzZhMpkEju1qtdtBFJHk7tAwGAzNmzKC7u3tI9w+Hw2zcuBG5XI5Op6O9fTW7du0Cdp9T1dfXU11dzYgR46m+9uuYTKZD2fx9mjp16hfeh8FgQKvVctpppx2CFgnDQQS9gnAAOp3uMwuOG41Gli1bdoRaJAiCIJxIysrKuPzyy/d7eyaTYfHixUewRQcmlytoGDsZe+WIId0/k0nj7m7HqRng3HPPRavVHuIWCgeiUqk4+eSTOfnkk6XrksmktG46FApJo/Xt7e2sXbuWTZs24XK5iMViUuIuh8PBWWedxdVXX01jYyM1NTWYTCZpJFmhUAzKti8Ih5sIegVBEARBEAThBBeJRHC5XPT09NDR0YHb7SYUCuH1eunu7qajo4OBgQFkMhlWq5Xa2lpqamo477zzMBgMFBUVUV1dTW1t7RdODCgIh5oIegVBEIRhlU/Sk81mgd1rXtVq9aBt8slvZDLZZ05nTKfTpNNpaR2gIAjCiSq/BjqfRCyTyRAMBtm1axft7e3s3LmTbdu24Xa7iUQi0rrrWCyGRqOhtLSUL33pS3z5y1+mtrYWg8GARqORslibzWZxnBWOCSLoFQRBEIbNwMAAzz77LIsWLaKnpweZTMaZZ57JbbfdxtixYwHo7u7mlVdeYc2aNRgMBr72ta9x9tlnS4lu9pRKpbj77rvZtGkTN9xwA1/5yleG3LYtW7ZgMpk+M+mUIAjCcMpkMoRCIUKhEH6/n1AoRCAQoL+/n97eXjo6Oujs7KStrY2Ojg4pCZ/NZsNoNFJYWEhTUxPV1dWMHTuWmpoaKisrxXpp4bgigl5BEARhWGSzWV588UXeeOMNpk+fjtPpJBQK8bvf/Y45c+bwxBNPoFQq+fWvf83GjRu54ooriEQi3H777dx5551cddVVe5VwaW9v509/+hM/+9nPaGpq+kLte+SRRzj55JO57rrrvtB+BEEQDoV0Ok1vby99fX309vbS1dUl/e7z+YhGowQCAdxuNy6Xi3Q6jcPhoL6+noqKCsaNG0dlZSVmsxm73U5JSQk2mw2r1SrW1wrHPRH0CoIgCMMik8mwZMkSzjnnHK666ippSnNpaSk//elP6evrIxAIsGzZMl588UUmTpxINpslFArx2muvccEFFwwqqREIBNiwYQMGg4GTTjpJKl8UDAZZt24dsLtWb1lZGTKZjFwuh8/nw+fz0dLSgtPppLy8HL1ej8vloqOjg+LiYrq6uigtLcXlcmGz2dDpdADEYjG8Xi+lpaXStECdTse6deuor6+ntLSU9vZ2tm/fTnV1NRUVFfssfZbJZOju7sZgMLBt2zbS6TSNjY2YTCY2b95MOBymrq6O0tJS1Go12WwWr9eL2+2mp6eHyspKHA4HBoOBeDxOf38/Wq2WDRs2oNFoqKmpoaysTJzUCsJRKJVKkUwmSSQSUo3hWCxGb28vbW1t9PT00N7eLtUXj0ajUrkmmUxGJBIhnU5jtVqpqalh0qRJ1NTUUFdXR2VlpVQGSq1Wo9FoxOitcMJSwu4v3B07dgx5J8lkErfbTSKRYPny5UP+Yt21axfhcJhNmzbtc9rawejq6iISiQzpvp/m9XrZuHHjkDMH7tq1i2g0yocffrjX+rSDlU6n6erqQq/Xs2HDhn0Wpj8Yzc3N+P1+li9fPqT7A7S2th7WAuLHKplMRjqdxu1273ebSCQiTjgF4VNkMhmlpaWsWrWKtWvXMmHCBNRqNRMnTuTPf/4zJSUlbNmyhUsuuYSGhgZkMhkKhQKj0Sit/93Te++9x3333UdbWxs//elPeeaZZ7BarfzmN79h586dqFQqstkst912G+effz5dXV3cfPPNeL1erFYrLpeLMWPGcO+99zJr1izWrFnD9u3bcbvd3H777VxzzTU88MADnHPOOQB89NFH/PKXv2ThwoW88847fPDBB1RWVvLRRx/xq1/9ihUrVvDMM8+gUqmIRqOMHj2amTNn4nQ6B7Xb4/EwZcoUpkyZQigUwuVyodPpmDp1Kh9++CFKpZKOjg4eeOABLr/8cjZu3Midd95JIpHAaDTS3d3NJZdcwsyZM/nPf/7Dtddey8iRI8lkMqTTacLhMHfffTcXXnjhEfm/CsMjP8K3L3uumReGRygUwufz0d/fj8/nw+v14vF48Hg89Pf3S7d3dXXR3d2Nz+dDqVRSXFyMw+HAbDbT2NiI1WqltLSU8vJyqqqqKC8vp7KyEpvNNuRzREE4ESg1Gg2JRIKZM2cOObjLZrN4PB6y2Syvv/76kD90yWQSn8/HtddeO+RF8clkkq6uriHd99P++c9/smPHjiEHK4lEgs7OTn70ox8N+TXJ5XJ4vV4+/PBDXn/99SHtAyAajdLT08P3vve9Ie8jmUzS398/5Psfr5RKJT09Pdx+++373aa/v5+LLrroCLZKEI5+SqWSG264gTvvvJMf//jHVFdX09TUxFe/+lVOOukktFotJ510EnV1ddIIqd/v57XXXmPatGmYzeZB+zv//PPJZDLcf//9PProo9TV1TFr1ixkMhlPPfUUOp2Ot99+m/vuu4/a2lqWLVuG3+/nsccew+Fw8O677zJ79mwKCgq4++67aWtrY8KECdx4441kMpnPfD7Lly/nhhtuYMGCBSQSCe644w6uu+46LrjgAvr6+pg1axbPPfcc99xzz1739Xq9OJ1Opk+fTm9vLzfeeCOrV6/msccew2w2M3PmTJYuXcrZZ5/Nm2++iVKplG576623eP7557n22muB3Wugp02bxk033UQul+Ovf/0rDz74ICNGjKCmpuYQ/OeEo0m+/Mvvf/97lixZss9t8jMkhMOvv7+fzs5Ouru7aW1tpauri46ODrxeL6lUCqPRiFKpJJFI4PF4GBgYwOVyoVKppDq2EydOxG63U1tbi91ux2KxYLFYMJlMWK1WabaJIAgHT1lfX8/mzZtFD6Bw0DQazXA34agyefJktm7dSi6X2+82Dz/8MJ2dnUewVYJwbDjjjDNYsmQJy5YtY926daxcuZLFixdTXl7Ok08+SUNDA2azmXA4zFtvvcX8+fM5++yzmT59Okrl4BU6FouFyspKtFottbW1ZLNZ3nnnHa699lpcLhcADQ0NdHd3s3btWr7//e9z8cUXEwwGaW5uZvPmzfh8PnK5HJWVlRgMBoqLi6msrKS9vf0zn0tNTQ1XX301VVVVPPbYY8Tjcaqrq2lpaQGgsbGRRYsW7TPoLS0t5bLLLqOqqgqr1Yrdbufiiy9m9OjR5HI5TjnlFFauXIlcLmfGjBm4XC78fj8ej4f169dL0yPzz/Hb3/42DocDgKuvvprHH3+cNWvWiKD3OCSTyZg/fz7pdHq/2yQSCaZOnXoEW3X8SKfTxGIxwuEwsViMSCRCNBqlv7+ftrY2abS2r6+Prq4ugsEg2WxWKuGj1+tRqVSYTCZCoRC9vb1otVrsdjsXXnghtbW11NXVMWLECKxWq1S/Nn8RBOHQUMpkMhHECMIXIJfLP3OWxKdPzgVB2L0m9q233mLKlClcdtllXHbZZUQiET7++GNuvfVW/vznP3PXXXfh9XqZO3cua9eu5ZprruG73/3uQa1Ly2azuN1uXnnlFZYuXSpdP2LECAoKCti8eTNz5sxBLpdTWlpKPB5HpVIdsANrT5/ezul0YjAYgN0j0j09Pfzyl78ctE1NTQ2pVGqv9isUikHLYGQy2aDjSn4NcjabZeXKlTz55JMYjUaKi4tJJpPkcjmpPXq9ftBIkFKpxGAwEA6HD+p5CccelUp1wM+EmPb62RKJBP39/bhcLlwul5T5uL+/n4GBAVKpFDKZDJlMRiqVIh6Pk0gkSCaTpFIpstksRqNROg7I5XLUajUlJSWUlZVRVlZGaWkpTqeT0tJS9Hr9cD9lQTihiDNxQRAEYVhkMhnuuusuHnjgAaZNm4ZMJqOgoIDx48czcuRIurq6CAQCzJ49G5/Px+9+9zvKy8sPuhNJqVRSVVXF9ddfz8UXX4xMJiOTybBx40bGjh3L7bffjk6n45577qG0tJR///vffPDBB9L99wwUZDIZcrmcVColXffp9ZNKpVK6j8PhoLy8nGeeeQatVksul6O3txe/3/+FOsGCwSBz587lpJNO4oYbbsBms/Gvf/2LNWvWSNsMDAwQCASkv5PJJJFIBKvVOuTHFYRj0Z41avM/o9EoHR0ddHd3097eTldXF11dXbhcLhKJBHq9HpvNRmFhIcXFxRQXF2O1WnG73VLyOL/fj1wux+l0Ul1djcPhoKKigpqaGml0t6CggIKCAvR6vej4FoSjgPgUCoIgCMNCq9Vy1lln8etf/5pUKsXo0aORy+WsWLGCzZs3c+utt7Jq1SreeustfvWrX7Fr1y527doFQElJCSNGjDjgyWS+pu/rr79OY2MjBQUFfPzxx/zpT3/iueeeo7+/H7vdTjgcpre3lz/84Q+43W46OjooLCxEpVKxc+dO1q1bR0VFBRUVFfzxj3/EZrMRiUR48cUX9/vYX/nKV3jppZd46aWX+NrXvkZ/fz8LFixg/PjxXHDBBUN+zdLpNH6/H4VCQSAQoK2tjWeffZZYLIbb7SaXy9Ha2srzzz8vBeqLFi2ipqaGcePGDflxBeFolcvliEQig+rT5i+dnZ14PB4CgYB0eyKRwGAwYDabMRqNGAwGxowZw+mnn45KpcLlcjEwMIDf76e9vZ1UKoXFYqG6uprTTz+d+vp6qY6tWFsrCMcOEfQKgiAIw0KpVDJjxgx++9vf8sADD6BQKNBqtRQVFfGTn/yESy+9lDfffBOdTsfvfve7QSOv5557LrfccsteJYB0Oh1VVVWoVCrUajU//vGPyeVyzJkzB6/Xi9lslhJZ3XjjjTz88MNcf/311NbWMmLECOrq6njmmWeYN28eX/3qV3nyySd5+umnmTdvHt/61re46667WLVqFZWVlZx//vls2LABlUqFxWLB4XBIQbjD4eDee+/l+eef55ZbbiEajXLmmWfywx/+cK/XQaVSUVNTI01nVigUOJ1OLBYLsHuU2Wq14nA4sFqtXH/99Tz99NOsWLGCyspKJk6cyNq1a3n33Xc588wzqampIRwOc//990vllmbNmkVlZeXh+lcKwmGXyWTwer309PTQ3d0t1al1uVx4vV7i8TharRadTodOp6OgoACn08no0aNRKBSEQiFCoRDBYJBgMIjP56OzsxO1Wi2NzFqtVioqKhg7diyFhYWUlpZSWFhIYWGhGK0VhGOcLHewi5cEQRiyOXPm0NbWxnXXXTfcTREOo3vuuYfvfve7XHXVVcPdlGNKNBolGAySSCSQyWTSKIxCoSAaje4z66xWq8VkMu21VjGVShEKhbBYLFISmEQigd/vJ5VKodPppNq+2WyWgYEB4vE4er0eg8FAIBBAJpNRXFxMKpWSyoYUFhaSSqWkk2udTofZbCYajWIymUgmkySTyUFtyuVyBINBwuEwcrkcs9m8z3V8+bq7ZrNZWlMcCATQaDSDagLn958/+U8mk+j1eoxGo1Sbd9OmTcycOZMFCxZI630NBoOY2nyCi8fjnHrqqdx7771UV1cftseJRqPccNNPuPhHD2GvHHHAbXO5HNlMhnQ6SSadIp1KkEmlSCSiLF+8EFVkFxdddJGU3bi7u5tMJoNOp8PhcGC32yktLcVms1FSUoLZbJY+Gx0dHbhcLnp6eujt7SUcDqPRaCgrK6O6upqqqioqKyupq6tDp9OhUqmkWrZDLTF5NMtkMmzYsEFaU7zn9R6Ph/b2dnp6ehgxYoQ04wZ2H5v6+/tZtmwZO3fuJJPJ0NjYSGNjI6NGjZI66oLBIKtWrRp0rC4pKeGss876zLZt3bqVzZs3S38rFApOO+00qc66IBwqottKEARBGFZ6vX6/SV0OdNu+qFQqCgsLB12n0WgoLS3da1u5XE5xcfGg60pKSgbt69N/f3o/+USQ+dGlPclkMsxm816llfbVjqKiokH3y4/y5u25f6VSOahdgHSCmA+4dTrdPp+zIBxpuVyOVDJOJOgjFvYTDQeIBL1EQgHCfg/hwACpZJx0Kkk6nSKdSDDg7qTErEYmkzF+/HhKSkowGAwolUpSqZRU6qerq4vm5mbi8TixWExak2u32ykrK2PcuHGUl5fjdDpxOBwnbEKv1atXc9111zFz5ky++c1vArsD2mXLlvHb3/6WgoICADZv3syTTz7J2WefTTabZcWKFbzyyivU1tYyefKH1KT6AAAgAElEQVRkysrK2LRpE3PmzGHy5MnceOONKBQKtm/fzi233EJNTY10TBwzZsxBBb3z5s1j7dq1VFRUAKBWq3E6nSLoFQ45EfQKwhHi9Xppbm4e7mYIh1E0Gh3uJggnuJKSEi644IK9pn0LQjqdZteuXYOSsR1q8XicaCTMx+//FaVKvTu4DQ4QjwRJxKPI5AqUSjUKlRqFQo2mwESB0Yqh0Ia2wIhGW4BKo2PHun9Sqg1jtVrZunUrb7/9Nm63m2w2S2FhIUVFRdhsNqqqqjjllFMwm81YLBapkymfRf1Et3LlSh599FGam5vZtGnToNsCgQAPPfQQF110ET/84Q+RyWTcfffdvPDCC5x11ln09fUxb948vve97zFlyhT+8pe/sGLFCr72ta9x5ZVX8tJLLzFhwgSmTJmCy+WiqamJ3/zmN9KskoPJsJ9MJmlvb2fGjBmcc845wO6OO3H8Eg4HEfQKwhFQW1vLu+++y9NPPz3cTREOI6PRKNVGFYTh0NjYyN133y3qewqDqNVqRo4cyeLFiz/XaGc6nf7c9WKdjjKy/ZtJkkMFWBSASQamgj22yoAshiwbh6CHdBDC7L4AGLNZ5PICWltbqa+vZ+rUqYwcOZLi4mIUCoWUpC1fQkjYN4fDwaWXXorX6+WFF14YdFtzczOBQIBzzz2Xvr4+VCoV9913HwUFBcjlcv7nf/6Hk046iZEjR3LllVcyZswYysvLufLKK7n00ku54ooreOWVVzjttNNYv3491dXVJJNJent7sVqt0oybP/3pT7zxxhvMnj2buro6PvzwQ+bPn8+dd96JTqfD6/XS2NhIb28vSqWS8vLygwqYBeHzEkGvIBwBV111lVjnKQjCESECXuHT5HI5f/nLXz7XfdLpNJFIZJ9r54VjQ01NDTU1NbhcLt59991Bt23btg2lUsnvf/971q1bh1KppLa2lhkzZlBTU8Pq1auZPn06Tz31FF/60pe47777UCqV/Pvf/6apqYnRo0fz9ttvE41G2bJlCzt27KC3t5d4PE4qleKWW27hvPPO4+KLL+a5557jt7/9LT/96U+ZM2cOo0ePpqGhgY8++ohAIMCDDz4IQG9vL6eeeio333wzdrt9OF4y4Tgmgl5BEARBEARhkEQigUajEQHvcSoQCLBp0yYqKiqYO3cusViMO+64g4ceeoh58+bh9XrJ5XKsX7+ehx9+GL1eTzQaRavVMmrUKFpbWzGbzeh0OgoLC/n2t7/NxRdfjFwu5+GHH2bWrFmMGzcOm83GnXfeyS233MKWLVvQarXcfPPNqNVqMpkMp512GtOnT6esrIyWlhauu+46Kioq+O///m/RgSccUuLdJAiCIAiCIEiy2SypVOq4zGQs7KZWq6msrOT2229n3LhxnH766dx5552sWrWKnTt34nA4aG5uJpFIIJfLSaVSvPnmmwwMDOB0Olm8eDFf/epXUavVPPHEE9xyyy00NTXR0NDA3Xffjc/nY/Xq1QCceuqpTJ06lTVr1nDFFf+/vTsPjOncHz/+npmYrLKQRWQTsseWhBBL1NpebrVKXUtblBZtLbUXVVRvLVWi9qXWpihVW5WLlNQa2xWJICKRkEX2bfY5vz98nV9DqLZXW/q8/jIz5zznmYk5cz7P85zPp5dcPq19+/Zs2LCBli1b4uPjQ7t27Rg0aBD79+9Hp9P9mR+P8AwSM72CIAiCIAiCTKfTUaNGDTHT9gyrW7cu1tbWctkhACcnJwwGA3q9ns6dO3Py5EkiIyOZMmUK9evX586dO2RkZDB//nwiIiLo2LEjpaWlbN++na5du8oZ49VqNZIkodfrAUhMTOTMmTP07NmTb775hg4dOuDp6clPP/0kHwvuJrGqUaMGWq32j/9AhGeeCHpBzmQobpwX/k7MZrN8YWNh8eCpQKPRUF5ejsFgwMbGBltb22q/IzqdDkmSUKvVD1wgGY1GDAYDSqVSLmPwuHQ6HWazGUtLy0e2q1arxfI7QRCEh5Ak6ZGztnq9HrPZLAc/kiSh0+moWbMmZrOZ0tJSNBoNFhYWcgmx6s65945To0aNB17X6/WYTCbUajUqlepX91+v11f7G2IwGDAajb+p3b+78PBw9Ho9R44cwcPDA4PBwNatWwkMDKRevXrUrVuXH3/8kZCQELy8vPDw8KB169bs378ftVrN+fPnyczMxMfHh88//5zKykr69+8PwI4dO3BwcCA8PJzy8nImT55M586deeuttxgxYgQLFy7k448/JikpiQ0bNhAcHIyTkxM5OTkcOnSI9u3bi1UGwv/cHxL0JiQkcOHCBbRaLbVr16Zjx47yaJBOp+PgwYPcuHEDS0tLgoKCaNGixS/+ZzebzSQmJnL27FkqKipwcHCgS5cu8o3ver2euLg4rl+/joWFBQEBAbRo0eKBOooAu3fvRqlU8uKLL4qTpvDMMxgMJCQkcOLECbKysnB3d2fw4MHUrl0buFus/uLFi+zevRudTodarcbe3h6z2UzXrl0JDg6W24mPj+fQoUNotVrc3d3p2rUrISEhwN2EFJs2bSIrKwtbW1siIyN5/vnnq/0O/pzJZOLUqVPs3r0brVaLh4cHXbp0oXHjxgDk5OSwZcsW0tLSsLKyomnTpvTs2VP8QAqCIFTj9u3bfPPNNwwZMqRKKZ+cnBwOHjxISkoKlZWVdOvWjY4dO2IwGFCpVJSXl7Nz505SUlIAcHZ2xmAwULduXfr06fPAIGh6ejoHDx6kV69ectkao9HIyZMnOXjwICUlJdStW5cePXrg5+f32P3PyMhgy5YtTJw4UX7ObDZz5swZ9u3bR2FhIe7u7vTo0YPAwMDf81E9s1QqFc7OzlVqntepU4eJEyeyePFiEhISKCwsJCcnh5iYGLlO+Icffsj69euxsbHBZDIRFxdHeno6RUVFvPrqq/j7+wPw3nvvERsby/Hjx1EoFGRmZjJmzBjq1KnDqlWrcHR0ZPDgwbi7uzNmzBgmTZrETz/9xPPPP89///tf3njjDQICAkhPT8fT05M+ffqI63Hhf041ffr06U/yACdPnuT999+nqKgIg8HAunXrKCwspEWLFqhUKpYuXcratWtxcXHh3LlzrFy5Eh8fH4KCgh7Z7pUrVxg4cCCFhYWYTCZiY2NJTEykR48eAKxcuZLly5fj6OjIxYsXWblyJe7u7jRs2PCBtnbt2kVJSQmRkZHiSyY88w4dOsRHH32Em5sbvr6+bN68mZycHDp27Ci/vmLFCtq3b0/nzp1p27YtPj4+XLlyhQ0bNtClSxesrKw4ceIEkydPpnHjxvj6+rJx40bi4uLo3bs3FhYWjB07lvj4eKKioqisrGT16tXUrVtXDpof5vjx44wbN47mzZvj7+/P119/zfHjx2ndujVqtZr58+ezY8cO2rRpg1qtZtmyZTg4ONCkSZM/4uMTBEF4auh0OubNm8eyZct48803sbW9WzaosLCQmTNnkpCQQHh4OIWFhaxbt47w8HDs7e2pqKhg3rx5aLVaunfvTvv27WnatCnW1tbExsai0+mqnHO1Wi3jx4/n6NGjdO3aVQ6arl69yoABA3B1daVt27bs37+fw4cP88orrzzW0unS0lLmzJnD8uXLmTBhgvx8YmIio0aNwtLSkk6dOhEXF0dcXBzt2rWT36Pw/9WoUYOGDRsSGBgoB75KpZKAgAAiIyPx9vamWbNm9OvXj/DwcHm/WrVqERUVhaOjIxqNRi4v9Morr9CoUSN5Rr9x48ZERUVRr149GjVqxL/+9S+io6NRKBTY2dnxwgsv4OPjA4CbmxutWrWiTp061K9fn9atWxMcHIyvry/PPfcc/fr1E6X/hCdDeoJMJpM0atQoafDgwZJOp5NMJpO0f/9+KSgoSEpJSZESEhKkRo0aScePH5fMZrNkNBqlYcOGSb1795by8/Ol27dvS4WFhXJ7Wq1WysrKkrRarTRt2jSpX79+UnFxsWQymaTvv/9eql+/vpSRkSGlp6dLTZs2lb799lvJZDJJGo1Gmjx5shQVFVVtP+/cuSPl5+dLZrNZys/Pl0pLS6U7d+5IFy5ckJKTkyWTySQVFxdLSUlJUkZGhqTRaOR9zWazdPv2beny5ctSYmJilf5KkiQVFxdLFy9elFJTU6WysjIpJyenyv63b9+WLly4ICUmJkplZWX/47+AIFSl1+ultm3bSsuWLZOMRqMkSZJ05MgR6Z133pFKSkqknJwcqX379lJ8fLwUFxcnvf7661KvXr2kefPmSatXr5YmT54sTZ48WTIajdKrr74qTZkyRW7n1KlTUmhoqHTkyBGppKRECgoKkr7++mvJbDZLBoNBGjFihNSnTx/JYDBIqampUmZmptyvwsJCKTk5WSouLpaGDx8ujRkzRtLr9ZIkSVJOTo7k5eUlffvtt1J2drYUHh4ut2s2m6UJEyZIPXr0kHQ63R//gQqCIPwF6fV6ad68eVJkZKQUGBgoubi4SLm5ufLrW7duldq0aSNlZGRIkvT/r9e2bt0qFRcXS7Nnz5ZGjx4tpaenS5988on06quvSsOGDZOWL18ubdiwQerWrZt09uxZqby8XJo0aZIUGRkpBQQESFFRUVJ6erp8nClTpkivvvqqlJ+fL0mSJF2+fFlq0qSJtH37dqmgoEC6fPmyVFlZKfc5NTVVys7OloxGozRlyhQpMjJS8vPzkxwdHeU2TSaTNG/ePKl9+/bydVNZWZnk7+8vrV279ol/tn9nZrP5z+6CIPxmT3R5s8FgIDg4mICAAHnpoclkkpMjSJJEy5YtCQkJQaFQoFKpCAkJYd++fWi1WrZv305KSgpz587FxsaGjRs3cunSJaZOnYq3tzcRERE4ODgAyPurVCoKCgqIjIykVatWKJVKrKys8PT0pKKiotp+xsbGYmFhwaBBg1izZg0mk4nr169z7do1KioqGDNmDImJiZw7dw5ra2siIiIYP348arWa7777jq+//hqdTkdBQQGNGzdmwoQJ+Pr6cvXqVRYuXMj58+dxdXXFy8sLvV7PW2+9RfPmzTl9+jRz586lsLAQg8FAdHQ0kyZNombNmk/yzyL8jRUWFnLnzh0aNGjA3r170ev1NGnShMWLF6NQKNi+fTuhoaE4OjoyfPhwXnrpJZo1a8asWbPkZUmTJ0/mzp07NG/eXF6xAWBtbY1KpcJoNGJtbU3t2rXJzMzEYDBQWFhIRkYGISEhGAwGYmNjSUlJYcGCBTg7OzNjxgyMRiMffvghoaGhNGzYUF46p1arUavV8v29Hh4eXL9+HaPRSFlZGRkZGbi7u4tVGoIgCP+nRo0aDB48mN69e5OQkMC0adOqvH7s2DGaNWtGYWEhcXFxeHt78+9//xtJkjCbzRw8eJBp06YRGxtLfHw8w4cPJzMzkzlz5vD5558TEhJCYmIiYWFhjB07luHDh3P69Gk+//zzKsc5c+YMLVu2lG+fCQgIwM3NjUuXLhEcHMzIkSMZOHAgffv25ezZs0yfPp1x48bRqVMnRo8ezdtvv01cXBwzZsyQ29RoNFy8eJGoqCj5PmRra2saN27MjRs35PuKhf89kT9DeJo90aDX0tKSAQMGYGFhwcWLF4mLi2Pbtm288cYbeHl5ySdZe3t7AMrKyti7dy+NGjWiVq1atGnThs2bN7NmzRqCgoJYuHAhCxYskOuBKZVKkpKS2L9/P3v37qVv3764u7vj4uLCrFmz5JNsXl4eJ06ceOjyx+vXr1OjRg2MRiNXr17l9OnTzJkzB19fXxYvXszHH3/MmDFjGDBgACdPnmTGjBm8/vrraLVaFixYwLhx4wgKCiIvL4+3336bTp06Ubt2bWJiYsjLy2PVqlUoFAoWLVrErl276NWrF7dv3+azzz6jUaNG9OnTh8rKSsaNG8eaNWsYOXKkyJgoPBGZmZnodDqWLl1KSUkJKpWKwsJC3n//ffr378+ZM2do3bo1O3fupHHjxowYMQJLS0sOHz6MWq3Gy8sLk8mERqPh3XfflQezysrKiI2NRZIkwsPDqVGjBsOGDePf//43p0+fJjs7GycnJwYNGoS1tTWDBw+mZ8+eLF++HF9fX+Lj41m0aBEuLi68+eabcmItvV7PypUrqVWrFoGBgTg4ODBw4EBmzJjBmTNnKCsrQ6vVMnPmTBH0CoIg/IyTkxNOTk5kZGQ8cH5MT0+XlySr1WqKiorw8PBgxowZ1KxZE6PRiKOjI99//z3z5s2jRYsWXL16lQMHDhAQEMCFCxfQaDQoFAqcnZ3lNu9XVlZGrVq15MdKpRInJyd5mWyPHj1YuHAh9evX58MPP6RJkyZERUUByO26uLhU6b/ZbKakpISwsLAq10q1a9emqKgIo9Eogl5BEB7wxCMrKysrLCwsyM/PJzMzE5VKRVJSEsXFxVhaWuLs7IzZbObcuXMMGjQIGxsbhgwZgrW1NeHh4UyZMoVFixYxfvx4evbsSefOneXZW7VaTW5uLnfu3AEgOTmZgoIC1Go1Li4u8nOjR4/m9u3bTJ48+bH63LlzZzp06EBQUBDt27fH0dGR7t27ExQURPPmzXFycqK8vBwnJyemTJlCu3btsLKyoqKigho1alBWVkZWVhZxcXGMHTuWhg0bEhoaygcffCDfa3L+/HmuXLlCYGAgt2/fpri4mLZt27Jnzx5u3br1ZP4Ywt9eZWUlt27dwt/fn71797Jv3z7eeecdPv/8c27cuCGP8icnJ9OpUycsLS0xGo1kZGTQsmVLLl68iLOzs5wQw8LCghs3bjBp0iR+/PFHFi5ciKOjI6WlpezatQtPT086d+7Miy++KCdTkSSJunXrMm/ePLZu3cqoUaOYOHEirVu3RqlUYm1tTY0aNbh+/TqTJk1i586dzJw5k4YNG1JWVsaePXtwcnKiW7dudO3aFaPRSGxs7ENXcgiCIAhVVVRUkJaWxieffMLevXv57rvvyMrKYsuWLZjNZpRKJVevXkWtVtOyZUsUCgU3btzAy8sLNzc3rl27hpeX1y8eR5IkJEmq8pxCoZCP8frrr9O8eXN69+6NjY0N48ePf6x7cu9v81671T0vCIIAf2DJog4dOtChQwcSEhIYOXIkhw8fpl+/fmi1WtatW8dXX31FREQEo0ePpl69evJ+nTp1wtbWlry8PF544YVq223fvj3Xr1+nf//+bN++nWHDhqHVavnqq69Yv349YWFhTJ8+Xc4y90vs7e3l0UOVSoWFhYWcKl+hUMjLO2xtbblz5w6jR4/GxsYGjUZDUVERcDepg8lkqrJU+d6ST4CbN29y+/ZtvvjiC3lE0mg04uzsLOqTCU+MnZ0dzs7OdOjQQc6i3KNHD2bPnk1KSgoBAQEkJydjZWVFYmIi3bp1Y/PmzaSmplKnTh2WLl3KsGHDqFmzJnq9nt27d7N69Wo8PT1Zvny5nCju1KlTXLlyhdjYWEJDQzGZTLi6urJgwQLefvttXFxc8PT0xMbGBoPBQNOmTeU+mkwm9u3bx9KlS3FxcWH+/PnyyH9qaioJCQksXbqUtm3bAtCwYUPGjh3Lyy+/XKUdQRAEoXq2trZERUXJ2Y5dXFyIiooiNTUVKysrlEolBQUFVFRUkJ6ejqWlJYsWLaJDhw5cunQJJycn+bz8KFZWVg9c05SVlclJjWxsbAgICKCkpIQ6derI10iPcm9wtLKyskqQW15eTt26dcVKOUEQqvVEg9709HTef/99Ro0axXPPPQdA/fr18fT0JD09HZ1Ox6effsrJkyeZN28eTZo0qVLOxGg08uWXX1K7dm0iIyNZvHgxYWFhKBQKXn31VV577TV69+6NQqGgfv36eHl5kZaWhslkYunSpezevZtp06bRqlWrKmna/1d2797NnDlz+Oabb3BxcUGj0XDz5k0AuXZoWVmZvH15eTl5eXnA3aLgoaGhxMTEyFnqiouL0Wg0eHt7/8/7Kghw934qOzs7uTY13L3PV6FQoFareemllxg5ciQvvfQS69at4/jx40RERGAwGJg1axYdOnSQbxO4N7M7YcIEOnToUGV0vqioCAsLC/kCRqVSVSlTZjAYWLx4MY0bN6Zly5Z88MEHLFmyBDc3Nw4dOsT8+fMZM2YM7du3r9JueXk5FhYWcmZQAG9vb/R6vRgsEgRBeEzh4eGkpKTIQaNOp6O0tBRLS0usrKxo0aIFRUVFhIeH079/f7y9vfHw8ODAgQOkp6czdOhQOafKo3h7e5OamopWq8XKyor8/Hzu3LkjT0L897//Zfv27cTExLBo0SK2bNlC3759H3m7ioWFBd7e3iQlJaHX67G2tkan03Ht2rXHKnn5V1RSUsI333zD5s2buXPnDjVr1uSFF15g1KhRD+R5MZlMHDlyhMLCQnr16lVte8XFxcTGxnLp0iUGDhxIZGTkb+qXyWQiPz8fBwcH+f5pQXhaPdGg19nZmdq1a7Np0yYCAgKoXbs2hw4dIi0tjcGDB3P69Gm+/fZbZs2ahZeXF4WFhcDdGVEHBweOHTvG+vXr+fTTT/Hw8ODNN9/kyy+/5K233sLDw4NvvvmGsLAwPD09iY+PJyUlhddee41Lly6xbt06xo0bR3BwMEVFRRQVFVVZ9vy/kJubC9ydPbOxseHcuXNcvXoVg8GAq6srDRs2ZN26dbi6umI2m1m9ejU6nQ5AnhE7e/YsQUFBFBcXs2jRIrRaLQsXLqy2CLsg/F7W1tZER0ezYcMGPD09UavVLFiwgODgYMLCwnBycqJfv35cuHCBoUOH4ufnR9OmTRkwYABnz57l/PnzckKqOXPmEBoaiouLC5cuXQLuBrfBwcE0b94cgGXLltGzZ0+MRiObN28mKCiIOnXqsGPHDn766SdWrlxJvXr1+Oc//8kXX3zByJEjWb16NX5+fjg7O5OUlATcTcpSv359/Pz8UKvVbNy4kTfeeAOTycTq1asJDQ2lQYMGf9rnKgiC8DTp1KkT27dv59tvv6VTp06cPHmSY8eOMW3aNOzt7RkyZAhLly4lJCSE9u3bExUVRa1atThx4gSXL1/m4MGD1K9f/xeXIvfo0YNJkyaxceNGWrVqxddff01lZSXPP/88hYWFfPzxx3To0IE+ffpgNBpZunQpoaGhhIWFPbRNS0tLoqOjGTt2LN988w0REREcOHCA8vJyeSn20+ZezpeePXvi4+NDamoqsbGxmEwmPvjggyqB/OXLl5k0aRK+vr4PDXoPHDjA0qVLmTZtmjzg/FsUFhYyZswYhg4dSnR09G9uRxD+Cp5o0GtnZ8ewYcN4//336dWrFzVr1iQ7O5uePXsSFRXFV199RVpaGrNmzZIT1wB07NiRoUOH8umnnxIdHS3X55w0aRKjR48mMDCQESNGMGzYMPr164eTkxM5OTl07dqVLl26sGHDBm7dusWiRYtYtmyZ3G5ISAhr1qz5n72/sLAwzGYzr776KnZ2dtSuXZuwsDBiYmJ46aWX+OCDD5g5cyZ9+/aVj1+vXj2USiXe3t4MHjyYmJgYtmzZQmVlJTVr1mTWrFmixpzwxKhUKsaOHcv06dMZPHgwcDcQXrRokTwr27NnTwIDA9m1axenTp1i69atXL9+neDgYPr27Uvjxo3Jz89HoVCQlpbGvHnz5PZtbGz46KOP8PPz46OPPmLJkiXs2bMHlUpFYGAgH374IdnZ2Xz//fcMHz5crvM3ceJEli1bRnJyMhUVFRQXF/PZZ5/J7To4OPDuu+/SpEkTZs6cydKlS3njjTewsLDA09OT6dOn/08HtARBEJ4VdnZ2BAQEVEnuFBYWxqhRo1i8eDHLly+ntLSUPn368OKLL6JQKKhXrx7jxo3j0KFDxMfHc+rUKfLy8tBoNPzzn//klVdeeeBaxc7OjgYNGlQJ0Dp06ECvXr2IiYlhyZIlWFhYMH78eDw8PNiyZQuWlpYMHToUa2trevbsyaVLl9i2bVuVoNfe3l5ehn1Pu3btGDJkCHPnzkWpVMq/I40bN35Cn+KTo9Fo+O677+jbty9jxoxBqVRiMpnw8PBg+fLl9OnTp8r7X7ZsGZIkPXJGu6CgADc3N3r37i0/J0kS6enpGAwGvLy8qqyshLuJI7OysuTrVLi74jIrK+tX58yQJIkbN27g7u6OtbW1vEy+Zs2aj7WaMTMzExsbG/m6JCcnB5VKVe3vvF6vJzs7Gw8PjyqxxM/74eDg8FhL54Vnm0L6A+761+v1ZGRkUF5ejr+/P7a2tnIiA7PZ/GCn/q/8kMlkkk9m9xiNRlQqFQqFAoPBwM2bNykvL8fT05NatWo9Vrv3u7etUqms8m9ATuzz8/1MJpP8uKSkhNTUVGxtbfH19aWyspLs7Gzq16/PzZs3cXV1RaPRoNFo0Ol0DBw4kMWLF9O8eXMkSaK8vJy0tDRcXV1xdXUVGWiFP8TPvyOP+l6Ul5eTlZWFq6srtWrVqnKvlMlkqrbt+78r904xSqVS3t9sNle5P/7etj//Dt7v5+eCn/f/5+0KgiAID7qXOOrn7l3flJWVYWFhIV+b3c9oNJKdnY1Go8HHx+eRK9Eedpx7vxc//725l+Tq59vffw32W9p92mi1Wtq0aUN4eLg8IGBjY4NOpyM1NZV69epha2uL0Whk5cqVpKWlyWWf1q9f/0B7c+fOZf78+ZSXlxMZGclnn32Gk5MTc+fO5cyZM2i1WurUqUNMTAyhoaFkZmby/vvvk5GRIScDGzBgAC+++CIvv/wyV69excnJiTfeeIORI0fSrl07vvzyS1q3bg3AihUr2LFjBz/88AMTJ04kLy8PSZK4du0aixYtori4mKlTp6LX6ykrK6N///6MGTPmgWXbu3fvZvr06XTr1o2DBw9SWFhI586d8fLyIjY2FoPBgKOjIwsXLqR58+acP3+emTNncuvWLfn/zb1KFKtXr2bRokW0adOGs2fPYjAYcHJy4rPPPnvkKgLh2faHJLJSq9XVJpH6pUiCYI8AABsnSURBVIvV6k5gPx/FqVGjRrVLGn/tRfDPt71/v+pOpD9/7ODgQEREhPzY0tISJycnSkpK+OKLLwAYPXo0Go2GtWvX4ufnJ/dZoVBQs2bNh5ZSEoQn5XG+I0qlEnt7e0JCQqp9/XEuMB62TXXHvrft47QrAl1BEITHV9358v7EnA9bFmxhYfFYmZofdZz7Z+DuP/6j9v+17T5trKyseO+991i4cCGvv/46ERERREZG0qxZM/z9/eUZ3ZMnT3LgwAGWLVvG5s2bH9pe7969yc3N5ejRo8ybN48GDRowZswYAFavXo0kSSxcuJBPP/2UdevWsWLFCgA2bNiAjY0N69evZ/v27bz88svMmjWLqVOn8tprr9G9e3esrKwwGAxVBqfNZjNGoxG4O0ASHx/PK6+8wuDBgykoKGDKlCmMHz+eZs2akZiYyOzZswkJCaFXr15V/v5ms5mCggIUCgVr1qzhwoULTJ06lYYNG7Jy5Up5pdr27dtp3rw5y5Ytw9nZmdmzZ6NUKlm0aBFfffUV/fv3x2w2k5WVhb29PRs2bKCyspIFCxYwZ84cNm7cKEpa/U2Jq8YnxN7enuHDh+Pg4MDUqVMZNmwYJSUlzJw5s0rNOkEQBEEQhD+DTqeTE28Kf55+/fqxZcsWRo0aRUFBAStWrKB///5Mnz6dyspKcnNzWbVqFa+99hru7u6PbKtevXo0aNAAR0dHmjVrxp07dzhz5gw9e/bEy8sLb29v+vXrR2JiIlevXmXIkCHMmTMHDw8PLC0tKSoqkktwhoeHy4Pf/v7+vzgoLUkSPj4+jB49mqioKBISEnBxcaF9+/Y4ODjQsmVLoqKi2Ldvn5zj5ufs7OwYPnw4wcHBtG7dmtq1a8uJuCIiImjUqBHl5eUAjB8/nilTpsifR1lZWZXksQ0aNGDEiBEEBgYSFhbG22+/TUpKCsnJyb/2zyM8I57+IbK/KIVCQUhICLNmzfqzuyIIgiAIglCFJEnodLoHlpkKf6z8/HzS0tJo2rQpgYGB9O3bF4PBwN69e3nvvfdo1KgR2dnZ5ObmYmVlxeHDh7l27Ro5OTkkJCQQGhr6yAolOTk55OXl8cknn1TZztfXl7KyMkpLS1mzZg0Gg0HOqaFUKn9TzeN794PXrVsXjUZDbm4uiYmJcm6be5o0aVJt+0qlssrE0P2PVSqVXH3i6tWrbN++XS6vlZeXV2VFgIODAx4eHvLjBg0aYDKZuHXrllhh+Tclgl5BEARBEIS/Gb1ej0qlemrvhX1WXL58mWnTprF06VKCg4OBu7fvdenSBWdnZzmJk6urK3v27AEgMTGR/Px8Dh48iJeX1yODXldXV1xcXPjoo4/kYE+v15OTk4OzszOjR4/Gy8uLSZMm4ezszLFjx/j4448f2eefB6ylpaVVXru3dFitVuPs7Ex4eLi8hBruljSUJOl3VSnR6XR88MEHdO3alWHDhuHs7Mz69evZtm2bvI1Go6G4uFgucVhYWIhSqaxS8lD4exFBryA8JSRJoqioiOLiYnQ6HZaWlnJmxPsZjUYqKyuxt7d/ZHslJSWUlZXh4uIiavAJgiD8BRkMBgoLCykrK8NgMGBjY/NAplqTyURpaSkmkwlLS8tHzt6azWby8/MpLCzEx8fnj3gLwiM0bNgQk8nE7NmzGTt2LHXq1MFsNnP+/Hk0Gg0hISG0a9dOnuEEWLJkCYmJiYwaNeoXf7v9/Pzw8vLiP//5D0FBQVhYWLBr1y7i4uKYPXu2nDDK3t6eoqIitm3bhkajoaKiAjs7O5RKJTk5OeTn52NlZYWlpSUXLlzA39+fvLw8Dh06VG0CSpVKRWRkpFwBokGDBhQVFfHJJ5/QuXNngoKCfvNnVlFRQUVFBf7+/tjY2JCRkcHhw4cpLy+XM03fuHGDbdu28c9//hOdTsfSpUtxcXGhadOmv/m4wtNNBL2C8BQwGo3ExcWxcuVKsrKy0Ol06PV6unfvzpAhQ6hfv36V7U+ePMkPP/zwyOX1N2/eZObMmZSXlzNhwoQqCdl+rYMHDxIcHFxlKZEgCILw++h0OtasWcOOHTsoKyujsrISlUrF0KFD+de//oWTkxM6nY6dO3eyZ88eJEnCysqKwYMH07Jly2rbjI+PZ9GiRTg6OvLxxx9XO3D6OAoLCzl//jytWrX6zW0I4OTkxMKFC5kzZ458z65KpUKr1TJp0iTatWuHlZVVleDW2dkZNze3h87w2tnZyeV9VCoVs2bNYt68eYwcOZLi4mIsLS2ZOnUq3t7evPzyy2zcuJEff/wRtVqNWq3GbDbz7bffMnLkSKKjo9m8eTN5eXkMHz6c119/nQULFrBt2zbs7e3x8fGR77N1dHSsUkrpueeeY9SoUUyfPh0rKysKCgpo2bIlnTt3fiA5mbW1NW5ubvL95RYWFri5uVV5346OjhiNRmrVqkXXrl1Zs2YN33//PdbW1tSuXZtz586xY8cO4O5M89GjRzly5AgZGRlYWVkxc+bMR86KC8+2P6RkkSAIv8/Jkyd588036dy5M/3798fW1pabN28ya9YsvLy8+PLLL7GxseHWrVukpKTw+eefU1xczLFjxx7a5pYtW1iyZAmffPIJTZo0eeSs8C9p164dY8eOpXv37r+5DUEQBKGqxYsXs2TJEoYMGcJzzz2HWq0mKSmJiRMnMmrUKN59911OnjzJO++8w4gRIwgICGDZsmVUVFSwbt06XF1dH2hz+PDhlJaWMnHiRIKCgh5Z7/VRTp8+zaRJk9i0aRN169b9vW/1by8/P58rV65QVlYmr+QKCAioNnN1bm4uFRUVDwx435OXl0dRUZFc31eSJO7cucONGzeorKykXr16+Pj4oFQqqays5NKlS1RUVODm5oa7uzvXrl3Dzs6OkJAQioqKuH37NjY2Nnh7e1NaWkpSUhIVFRXUrVsXZ2dnSktLCQwMJDMzE5PJRL169eS+mM1mUlJSyMvLw9LSkpCQEBwcHB7oc3FxMZmZmTRs2BCFQoFeryc1NRUvLy955cLNmzflZFklJSUkJyej0+nw8vKidu3aXLx4EQ8PDw4dOkRsbCyxsbHcunWLkpISGjRoQL169UTStr8xMdMrCH9xJpOJqVOn0rZtW2bNmiWf/ENDQ7G3t2f69Omkp6fj6enJtGnTuHjxInfu3HnkRcjVq1c5ffo0cHe03srKCrPZTFpaGocPH0an09GqVSvCwsJQKpUYDAauX79OVlYWGRkZ+Pn5ERoaKl+A3bp1i7Nnz9KgQQO8vLxITk4mIiJCvrcnKyuLkpIS/P39ycjIQK1WU1RURFJSEl26dMHGxoZTp06RlJRErVq16NChQ7UZKouKisjIyMDOzo74+Hjs7e2JiIjA2tqauLg4FAoFgYGBBAUFYWVlhU6n49q1a6SlpVFQUEDDhg3x8/PDycmJW7duUVhYCNwdVKhVqxZBQUH4+fn9rnuNBEEQ/heSkpJYsWIFo0aNYtiwYfLzjRo1Qq/Xs2/fPrKzszl48CBdu3Zl4MCBWFlZ4e/vT3R0NBcvXqRTp07yfnq9nqSkJK5cuUKdOnXkMjM6nY5Lly5x4sQJbGxsaNOmDQEBAcDdGrKXLl0iKyuL0tJSGjZsSGBgIBUVFZw8eZKsrCzi4+OJjo5GpVJx+/ZtQkJC5ED60qVL2NnZUa9ePc6dO4ebmxvp6enk5ubyj3/8g4qKCuLj48nIyMDf35+2bdtWOwCblpaGRqPBZDJx6tQpfH19CQsLIy8vj5MnT+Ls7ExwcDC+vr6oVCpKS0u5cuUKN2/eRK/XExQURMOGDalRowZJSUkolUoqKipISEjA19cXPz8//Pz8nuSf8xc5Ozvj7Oz8WNu6ubk98nVXV9cqAx4KheKB5+6xsbEhMjKyynM/f+zk5ISTk1OVx23atKmy/b3f6+pKWymVSkJCQh5a/vAeR0fHKvfbqtXqB/bx9vaW/+3g4EBUVFSV16OjowE4dOgQSqWSunXrigEZQaaaPn369D+7E4IgPNzNmzdZsmQJgwYNemAJsru7Oy+88ALu7u7Y2NgQFRVFr169sLS0JD09nYEDB1bb5ubNm/n666/JzMzk+vXr9OjRg1OnTvHRRx9x+/ZtUlNTiY2Nxd/fH19fX+Lj45kyZQqpqalkZ2cTGxtLeno6/v7+xMTEcObMGXJzc1Gr1dja2jJgwAD69u2Lra0tAOvXryc2Npbo6GhiYmLYt28fmzZtIiMjgzZt2rBkyRK2bNlCRUUFP/zwA6dOneL5559/YAbi2LFjDB48mPPnz1NQUMChQ4fYvXs3Fy5c4PTp09y4cYM1a9bg6+tLgwYN2LVrF7NmzSI7O5tbt27JGSqbN2/O119/zbhx47h48aK8TG/Tpk24ubkRGBgoRoMFQfhTHThwgL179/LFF188sCQzODiYtm3b4urqir29PS1atJAD2fj4eM6fP8/AgQOpXbu2vE9ZWRnLly/n4MGD5ObmYjKZiIiIYOvWrcTExFBRUcHRo0f54Ycf6Ny5MzVr1mTNmjUsXLiQvLw8rl27xpo1a6hVqxYGg4HPPvuMjIwMbt68SUBAAFeuXGHu3Lk8//zz8uDsiBEjyM3NJTo6mt69e3Pu3Dm++uorLCwsqFevHp988glHjhyhsrKSjRs3otPpCA8Pf6CO6vz58/n8889JSUkhNzeXb7/9lnPnzvHdd99x8+ZNeVlrREQE9vb2xMTEsHbtWvLz87l48SJr164lICCA+vXrM2rUKFavXk1KSgpFRUUcPHiQnTt30qpVK1FS8hmh0Wiws7N7ICgW/t7ETK8g/MVlZmZSo0aNKsuF7qlRowZ16tSRH98bxXVycnpk0Pb666+j1WqJi4tj/vz5lJeXM2/ePLp160a/fv0wm81s3LiRmJgYgoKC2Lt3L82bN2fMmDGoVCrWr1/PV199xSeffMJnn33GmTNneO+99+jduzeXL1/+xfd04MABVqxYQbNmzThz5gyHDx9mwYIFBAcHk5mZyZgxY9iyZQuDBw9+YN/i4mJ69OjByy+/THJyMgMGDCA8PJyZM2diNBoZO3YsJ06coFWrVhw8eJAXX3yRQYMGAXcvnBISEuT7j3Jzc1m4cCGtWrXCYDCwePFi1q5dS8uWLat8roIgCH+0vLw87OzsqgSu96jVankGq0WLFgAkJCQwc+ZMrl69yty5cx/4zbC3t2fChAkkJSURFBTEBx98QHZ2NqtWrWLKlCm0b9+esrIyJk2axIIFC5g+fTp79uzhzTffpHv37mg0GqZOncqxY8f4/PPP+fDDD5k6dar8O3Evs/DDFBcXc+PGDebPn09AQAALFy6kuLiYVatWYWdnx7Fjx5gxYwYdO3akWbNmVfaVJAm9Xs+7775LaGgo3377LePHj2flypW0a9cOnU5H+/btSUlJwdXVlaNHjzJixAiee+45KioqGDJkCCdPnqRTp06YzWYsLS0ZN24cgYGBlJaW8t5777Fy5Upmz55d7XJi4enSsmXL35WnRHg2iaBXEP7iVCoVkiRhMpkeeM1sNlNSUoKVldWvSiRib2+Pvb09lpaWuLq6cu3aNZKSknjuuefYt28fcHc51JUrV8jIyODf//43qampnD9/nrS0NPbu3YtGo0GpVOLs7IyFhQWOjo7V3qdTnbZt29K5c2esrKzYu3cvFhYWXL9+nevXr8v92717d7VBb/369WndujX29vZ4e3vj4OBAx44dcXNzQ6PR4O7ujk6nw8HBgZiYGFJSUjh+/DiZmZns378fDw8P+bNs1aoVERER8qxEz549+eqrr8jMzBRBryAIfyqFQiEvQb6f0WikpKQEBwcHOYtz3bp1GTRoEMnJyUyZMgVbW9sqy5uVSiV2dnZYW1tjb29PrVq12LRpEwaDgZycHDlodXFx4fjx45SWlrJ161YuXbrE8ePHSU5O5sSJE7Ro0QIrKytq1aqFhYUFLi4uj/X7o1Ao6N69O61bt+b27ducPn0af39/fvzxR+BuRl6lUkl8fPwDQS9AYGCgPHPn7u6Om5sbHTt2xN7eHoPBgJOTE0ajEVdXV7Zu3crly5c5evQo//3vf7l8+bKctVepVNKmTRsiIiJQKBTUrFmTnj17smrVKm7fvo2np+ev+CsJf0UWFhZVspsLAoigVxD+8ry9vTEYDKSlpdG6desqr2VnZzN37lx69epF27Ztf/MxTCYTOp2OtLQ0ioqK5Of79euHm5sbK1eu5McffyQsLAxLS0uCgoLIz8//zcfz9PSUa0NqNBq0Wi2JiYny635+fg8tpaFSqR6Yxb5/KRzcXcq3evVqTpw4QfPmzbG2tqZx48YUFBTI29ja2lapUVmjRg1UKlWV0hCCIAh/hjp16qDT6cjOzn4gx8G92dwPP/yQW7du0aBBAzw8PHjllVfo0aMHJ06c4MiRI7Rr167K+VGn01U552m1WjQaDdeuXZO3s7S0pG/fvhgMBqZPn052djYhISHY2dnh7+//m9+PtbU1Li4ucjBvMBi4c+dOlXN/u3btCA0NrXb/+8/z1Z334W6eiokTJ6JUKvH398fBwaFKIKtUKrGxsanyO1KrVi10Oh1arfY3vz9BEP7aRNArCH9xHh4eNG7cmO3btxMdHS0Hg5IkcfLkSfbv38+YMWN+1zGcnJzw9fWlT58+tGrVCoCMjAx2796NjY0NMTExjBgxgrfeekte3vywzNAqlQqlUolGo5H7mZaWVmWbn4/ANmvWjOLiYiZMmICtrS1Go5HvvvsOOzu73/WeMjIyWLt2LbNnz6Zz586oVCrmzZtXJei9fv06JSUlcpKOW7duYWFh8bsyWQuCIPwvREZG4u7uzuLFi5k8ebKcI8FsNrNt2zYKCgpQqVTyfbTvvvsuSqUSo9FY7eCgwWBAkqQqzzdp0gQ3NzfeeustfH19kSSJI0eOcPPmTa5evcr3338v34piMBi4cuUKlZWVAA+0r1ar0Wq18uy0Vqvl1q1bNGrUSN7+3tJhZ2dnvL29CQ4OZsyYMSiVSgoLC/nmm29+d+KhhIQEEhISWLlyJWFhYZhMJvbv3y+/bjQa5SzGNjY2mEwmLl26hLOzc7VLyQVBeDaIoFcQ/uIUCgUTJkxg6NChjB8/nrfffhtbW1vS0tJYtWoVffr0qTbT8a/h5eVFdHQ0X3zxBXq9HpPJRGxsLDY2NlhbW6NQKLhx4wZnzpwhMzOTrVu3UlxcTFpaGsHBwSiVSo4dO4a3tzd169alTp06xMTE0K1bN27fvs3BgwflC5/79erVi40bN7Jo0SLatWvH5cuXWb16NbNnz/5d70mpVKJUKrl8+TKOjo6kp6ezY8cOrKysyMnJAeDy5cssW7aMbt26odVqmT17NtHR0VUyRAqCIPwZGjRowLvvvsusWbPQ6XS8+OKLKBQKLl26xLZt25gxYwYeHh5ER0ezadMm7O3t8fLy4uzZs6SkpPDOO+88MMtraWlZJVht2bIldnZ2LF26lB49epCXl8eCBQsYMGAA1tbWGAwGkpKSUCgUJCQkcPjwYXx9fcnPz8fKygqtVsvOnTvp2rUrHh4e5OXlsWXLFiIiIjh+/Di5ubnVvjdbW1t69+7NrFmz8PHxwdPTk0OHDnHo0CG6dev2uz43CwsLysvLSU5ORqPRyJUBnJ2d5ZVMR48eZdmyZURFRXH9+nU2btzIiBEjqmQpFgTh2SKyNwvCU8DNzY2mTZty+vRplixZwq5duzh16hQvv/wy77zzjnxP6j3p6ekUFhY+sm5uRkYGpaWldOzYUa7Hl5WVxYYNG9izZw++vr5MmjSJOnXq4OzszM6dO9m7dy86nY7nn3+eY8eOoVaradu2LQqFgvPnz6PVavnHP/6BhYUFK1asYOfOnZSWltKjRw9UKhXR0dGkpqbi4OBAixYtUCqV1KxZEx8fH44ePcqmTZtITk5m0KBBvPLKK1WW4QHcuXOHtLQ0unTpgr29PTqdjtOnT/Pcc89Rt25dJEkiOTkZBwcHOnXqRM2aNdm4cSOHDx+msrKSrl278v333+Pr64vZbJZrC27atInNmzfTpk0bxo0bV21ZB0EQhD9acHAwbm5ubNu2ja+//ppdu3aRnJzM1KlT6datG5aWlgQEBKBQKPjuu+/YuHEjycnJTJgwga5du8pBr9lsRqPRYGtry9mzZ/H09KRZs2ZYWloSHBzMuXPn2LhxI0ePHqV79+68/fbbuLu7YzAYWLNmDSdOnMDOzo6wsDDi4uIICwsjPDyc/Px8Tpw4gY+PD02bNqWoqIjly5dz+PBhPD098fPzo379+oSHh3PkyBGaNWsml0Py8fHB3t6eLVu2sHXrVjQaDR988AFNmjR5IJnU5cuXUalUdOzYEbib5Cs1NZUePXqgVquRJImffvqJ5s2bExkZSUFBAZs2beLixYvUrl2byMhIdu3aRcuWLTl//jxeXl6o1Wo2btzIf/7zH958800GDhz4m2sWC4Lw16eQJEn6szshCMLj+/lX9kmU1bnX/v1t3//8/cvkHred33Ls3+Nh/V62bBk//vgjy5cvl2sDijJFgiD8Fd1/qVbduepRvw0ajQZJkh4ofVTd/r/23F/d44f18dcc9/d6WL/79++Pv78/06dPf2LHFgThr0csbxaEp8yT/nF+WPv3P/9L/fgt/XwS7+1x+i0ueARB+Ct7nHPUw7aRJAmdTvfAiqDH3f+XzqG/9rfhcY/7e4lzvyAIPyeCXkEQ/pY6duxIw4YN5eQwgiAIzyK9Xo+FhcUDt4v8XQ0fPlwkKxSEvyGxvFkQBEEQBOEZVVpairW19UNL/AiCIPwdKH95E0EQBEEQBOFpc69MkQh4BUH4uxNBryAIgiAIwjPoXpkiQRCEvzsR9AqCIAiCIDxjzGYzRqNRBL2CIAiIoFcQBEEQBOGZo9VqUavVIkOxIAgCIugVBEEQBEF4pkiShF6vF7O8giAI/0cEvYIgCIIgCM8QUaZIEAShKhH0CoIgCIIgPENEAitBEISqRNArCIIgCILwjBBligRBEB4kgl5BEARBEIRnhE6nw8rK6s/uhiAIwl+KCHoFQRAEQRCeASaTCaPRiFqt/rO7IgiC8Jcigl5BEARBEIRngE6nE2WKBEEQqiGCXkEQBEEQhKecKFMkCILwcCLoFQRBEARBeMqJMkWCIAgPJ4JeQRAEQRCEp5xWqxWzvIIgCA8hgl5BEARBEISnmMFgABBligRBEB5CBL2CIAiCIAhPMVGmSBAE4dFE0CsIgiAIgvCUEmWKBEEQfpkIegVBEARBEJ5SokyRIAjCL/t/BJxEKE1pTyEAAAAASUVORK5CYII=)

LeNet是由Yann LeCun提出的一种卷积神经网络架构，在1998年被用于MNIST手写数字分类数据集上，取得了很好的结果。LeNet是目前卷积神经网络中的经典架构，被广泛用于数据集较小的分类任务中。

LeNet的架构有两个卷积层和两个全连接层，其中卷积层和全连接层的输出通过Sigmoid层进行非线性变换。同时，LeNet还使用了池化层进行下采样。

LeNet的输入数据是32x32的灰度图像。它使用了许多尺寸为5x5的卷积核，以及许多尺寸为2x2的最大池化层。

- 第一层卷积，20个卷积核，尺寸为5x5。
- 第二层卷积，50个卷积核，尺寸为5x5。
- 第一层全连接，500个神经元。
- 第二层全连接，10个神经元（对应到MNIST数据集的10个类别）。

LeNet的输入是32x32的灰度图像，输出是10个类别的概率分布。LeNet通过两个卷积层和两个全连接层进行前向传播，最后使用Softmax层对输出进行归一化，得到每个类别的概率。

***

**大纲**

1. 导入必要的库，如pytorch、numpy、matplotlib等。
2. 加载MNIST数据集，并进行预处理。这包括将数据转换为tensor、分割数据集、归一化数据等操作。
3. 定义深度学习模型。这可能包括定义网络架构、初始化权重等操作。
4. 定义损失函数和优化器。这可能包括选择合适的损失函数和优化器以及设置超参数。
5. 开始训练模型。这可能包括将数据输入模型、计算损失和梯度、更新参数等操作。
6. 评估模型性能。这可能包括在测试集上测试模型、计算准确率等操作。
7. 保存模型。这可能包括使用torch.save函数将模型保存到磁盘中。
7. 需要定义可视化函数来可视化模型的训练

***

#### 使用GPU

安装完成CUDA11.7和其对应的pytorch1.13.1，CUDNN11.X后我们可以通过调用GPU来加快运算速度

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

需要在GPU上计算的数据都需要从CPU上转移到GPU

比如我们需要转移已经定义好的卷积网络模型model，和每次从数据加载器从处理得到的每一轮batch中的（data,target)

```
model = ConvNet()
model.to(device)
# 训练集
  for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
# 测试集
 for data,target in test_loader:
            data, target = data.to(device), target.to(device)
```

**补充**：遇到的小问题1：

如果采用了matplotlib作为可视化工具，我一般是采用创建一个losses[]和一个accuracys[]，来记入每次小循环和大循环中的损失函数和正确率的变化,然后通过plt.plot和plt.show来进行可视化图像。

但是似乎plt.plot在可视化的时候需要调用numpy，如果这个时候张量内存保存在GPU上而不是CPU，无法在CPU上进行numpy操作，会报以下错误

`can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`

这时候可以通过转移张量来实现可视化

```
losses_cpu = torch.tensor(losss, device="cpu")
accuracys = torch.tensor(accuracys, device="cpu")
```

```
plt.plot(losses_cpu)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.show()

plt.plot(accuracys)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.show()

```



***



#### **图像的像素值**

图像的像素值是指图像中每个像素点的颜色信息。在计算机中，图像的像素值是以数值的形式存储的。不同的图像格式使用的数值类型可能不同，例如在 8 位 RGB 图像中，每个像素值可以是 0 到 255 之间的整数。在图像处理中，通常会对像素值进行归一化处理，将像素值转换为 0 到 1 之间的浮点数，方便进行数学计算。

***

#### **图像的预处理张量转换**

torchvision.transforms.ToTensor函数是pytorch中的一种数据预处理方式，用于将图像数据转换为张量（Tensor）。

- 对于图像数据，它会把numpy数组的值归一化到[0, 1]范围内，并转换为float类型。

  - 对于数据归一化起到什么作用？

    归一化是指将数据转换为0到1的范围内的值。这通常是通过将数据减去最小值并将其除以最大值与最小值之差来实现的。这种归一化方法称为min-max归一化。

    归一化通常用于将不同尺度的数据转换为相同尺度，以便进行比较或使用相同的工具进行处理。在机器学习中，归一化也可以帮助模型更快地收敛并提高准确率。

- 对于标签数据，它会把numpy数组转换为long类型。

对于一个灰度图，这个函数会将每张图像的像素值从[0, 255]范围内的整数转换为[0.0, 1.0]范围内的浮点数，并将每个像素值单独放在一个位置上。一张灰度图像可以被转换为一个2维张量，而一张彩色图像则可以被转换为一个3维张量，分别对应着红、绿、蓝三种颜色通道。例如，如果你有一个大小为(3, 256, 256)的numpy数组img，它表示三通道的RGB图像，每个像素的值在[0, 255]范围内。那么使用`torchvision.transforms.ToTensor`转换后，它会变成一个大小为(3, 256, 256)的torch tensor，每个像素的值在[0, 1]范围内，类型为float。

这个函数在深度学习中常被用于加载图像数据集，例如MNIST数据集，因为张量比较容易被神经网络处理，而图像数据本身则是像素矩阵的形式。

```
from torchvision import transforms

to_tensor = transforms.ToTensor()
input_data = to_tensor(input_data)
```

这里 input_data 可以是任意的 Python 对象，例如 NumPy 数组、PIL 图片等。



***

#### **Pytorch内置的数据集**

Pytorch自己有内置一定的常用数据集，可以通过torchvision库中网上下载

```
import torchvision.datasets as datasets
```

若我们需要加载MNIST数据集我们可以通过以下方式导出

```
train_dataset = torchvision.datasets.MNIST(
    root= './dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
```

```
# dataset类型的函数有以下参数：
# root:表示下载数据集保存的路径，训练集和测试集都会下载到这个文件夹内部
# train：布尔类型，表示是否为训练集
# transform；表示如何对数据转换,这里使用的transform.ToTensor函数可以使得图像转换成张量
# download:布尔类型
```





***



#### **数据加载器**

**torch.utils.data.DataLoader**

**util n.常用工具，存放类函数**

数据加载器是用来将数据集的数据加载到模型训练过程中的，它可以将数据集的数据分成多个批次，并且在每个训练周期（epoch）开始时打乱数据，每次在训练时取一个批次的数据进行训练。数据加载器还可以在每个批次的数据进行取样时进行数据增强，这对于训练模型有很大的帮助。

1. 用DataLoader可以实例化train_loader和test_loader，实现对数据集的mini_batch打乱和划分。所需要的参数如下

- `dataset`：dataset为已经处理好的数据集，可以导入模型使用。这里起到一个随机打乱的作用

- `batch_size`：超参数，决定了每一个min_batch批次训练的大小

  - 在数据加载器中和四维张量上我们都遇到了batch_size这个超参数，两者的差别在什么地方？

    结论：

    一般来说我们会把数据加载器中的超参数batch_size的大小和四维张量的第一维大小设置一样，这样使得代码更有可读性。虽然在层次和顺序操作问题上他们不一样，但是在目的上他们是一样的。

    展开：

    在使用图像分类数据集时，我们通常会把图像数据转换成张量的形式，并将其存储在四维张量中。四维张量的第一维（即第一个方括号内的数字）表示批量大小，即每次迭代时加载的图像数量。而数据加载器中的batch_size参数则是指每次迭代时从数据集中取出的样本数量。

    在 PyTorch 中，数据加载器返回的是一个迭代器，每次迭代时返回一个批次的数据。因此，我们需要使用四维张量来表示一个批次的数据，其中第一维表示批次中的样本数。数据加载器的 batch_size 是用来控制每次迭代返回的数据量的，而四维张量的第一维是用来表示每个批次中的样本数的。这两者是有关联的，但是并不完全相同。

    举个例子，如果我们的数据集中有1000个样本，我们将数据加载器的batch_size设置为64，那么每次迭代时，数据加载器就会从数据集中取出64个样本。如果我们将四维张量的第一维设置为32，那么每次迭代时，我们就会在四维张量中加载32张图像。所以，这两者是有区别的，但是目的是一样的，都是用来划分数据集的。

    果你设置的batch_size大于四维张量的第一维，那么在一个batch中会进行多次处理。比如说，假设你的四维张量的第一维是10，而你设置的batch_size是20，那么在一个batch中会进行两次处理，每次处理10个样本。

    在MNIST这题中我采用了这种表达方式batch_size*(1,hight,wide)来表示数据的维度。

    

- `shuffle`：这个参数决定了是否要对数据集划分后的结果进行随机打乱。如果这个参数为 True，数据加载器会在每一个 epoch 开始时将数据集中的数据打乱。这可以帮助避免数据集中出现的任何模式对模型造成影响。

- 一般采取对训练集shuffle的参数设为True,测试集设置为Flase。如果数据的顺序不同，则模型可能会看到不同的数据顺序，从而导致不同的学习和更新。相反，如果在测试模型时按顺序遍历数据，则可以保证模型在测试过程中看到的数据是一致的，这样可以使测试结果更加可靠。

  - 训练集不设置shuffle导致的可能后果--泛化能力差。

    ①固定的数据集顺序，严重限制了梯度优化方向的可选择性，导致收敛点选择空间严重变少，容易导致过拟合。模型可能会记住数据路线。

    ②如果数据是排序过的，比如按类别排序，会导致模型一会儿过拟合这个类，一会儿过拟合那个类，这一方面会使得训练过程的loss周期性震荡；另一方面，在训练结束时，模型总是对最近训练的那类数据过拟合而导致泛化能力差。

  - 什么时候考虑不用shuffle？

    序列数据需要从样本的先后顺序中获取顺序特征，一般不用shuffle。

    如果我们想让模型学会某种次序关系或者我们希望模型对某部分数据表现的更好一点，那么我们则要根据自己的目的来决定数据的顺序，并决定是局部shuffle还是完全不shuffle。比如，对于时间序列数据，根据过去的数据预测未来的数据，我们则希望越近期的数据，模型给予更高的关注度，一种方式就是将近期的数据放在后面，那么等模型训练完的时候，最近见过的数据就是近期的数据，自然对近期数据有更高的关注，这样在预测未来数据的时候，近期的数据能发挥更大的作用。

    故我们要灵活的考虑是局部随机打乱还是，还是全局打乱

  - 数据集过大可能导致无法shuffle情况

    拆分数据集,或者进行部分shuffle

- `num_workers`：num_workers是在使用数据加载器时可以使用的参数。这个参数是用来指定数据加载器使用的进程数的。num_workers的值越大，数据加载就会越快。这个参数设定数据加载器使用多少个子进程来加载数据。可以加快数据加载的速度，但是会占用更多的 CPU 内存。

- 使用多进程加载数据时，要注意的是，由于在每个进程中的内存是独立的，所以如果每个进程加载的数据都要在内存中保存，这可能会导致内存不足的问题。因此，在实际使用中，应该根据需要适当调整num_workers的值。

```
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size =BATCH_SIZE,
    shuffle=True
)
# 测试集的shuffle=False选择不打乱，保留数据集的顺序
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size =BATCH_SIZE,
    shuffle=False
)
```

2.DtaLoader的返回值是一个迭代器，迭代器中含有多个bitch

可以通过调用来获得每一个batch内部的数据。训练批数据是以元组（data,target)的形式保存的，其中包含了训练数据的输入和标签。在图像分类问题中，data是图像的张量，这里target为图像标签。

- ```
  for data,target in test_loader:
  ```

- `enumerate()`函数是一个内置函数，它接受一个可迭代的对象作为输入并返回一个元组，其中第一个元素是索引，第二个元素是迭代的元素。这对于遍历列表或序列的时候获取索引和元素的值很有用。

```
for batch_idx, (data, target) in enumerate(train_loader，start=1):

```

`enumerate()`函数还可以接受一个可选的起始索引作为输入，默认为0。利用batch_idx我们可以在后面打印出程序运行的进程

- 元组的补充：

***

#### **定义卷积神经网络ConvNet（nn.Module）**

##### 基础框架Self-forward

`Class Convnet(nn.Moudule)`继承了torch.Mouduled的定义后要求实现两个部分

- `self.`定义所需要的卷积层，池化层，全连接层。在这里可以考虑把卷积层和池化层还有激活函数统统放到一个nn.Sequential内部进行封装。
  - LeNet在进入卷积层->激活函数->最大池化层->--->全连接层1->激活函数->全连接层2->F.log_softmax

- `def forward(self,x):`nn.Moudule定义卷积网络中必须定义前项传播函数（未识别到前项传播会报错）。

  - shift+tab可以集体缩进

  ```
  class ConvNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Sequential(
              nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=2),
              # output.shape = (20,28,28)
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2,stride=2)
          )
          self.conv2 = nn.Sequential(
              nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2,stride=2)
          )
          # 全连接层
          self.fc1 = nn.Linear(50*7*7, 500)
          self.fc2 = nn.Linear(500, 10)
  ```

##### **conv2d二维卷积层**

**nn.conv2d(in_channels,out_channels,kernel_size)**

```
nn.conv2d(in_channels,out_channels,kernel_size)
```

1. nn.Conv2d()是PyTorch中的二维卷积层，可以将其用于卷积神经网络的构建。它的主要参数如下

   - in_channels：输入的通道数，即输入的特征图的数量。

   - out_channels：输出的通道数，即输出的特征图的数量。

   - kernel_size：卷积核的尺寸，可以是一个整数或一个二元元组。

   - stride：卷积的步幅，可以是一个整数或一个二元元组。

     卷积层的默认步长stride为1；最大池化层的默认步长和其对应的kerner_sieze的大小一样，这样，最大池化层会将输入的图像缩小为原来的一半。

     在设计卷积神经网络时，通常会在每个卷积层之间使用步幅为 2 的卷积层，这样可以使得网络在计算时需要较少的参数，并且可以降低过拟合的风险。

   - padding：填充的尺寸，可以是一个整数或一个二元元组。这里采取了padding=kernel_sieze-1使得图片的大小维度不改变。

   - dilation：卷积核的膨胀系数，可以是一个整数或一个二元元组。

   - groups：卷积分组的数量。

   - bias：是否使用偏置。

2. 经过一次二维卷积层后的输入数据张量维度减少（kerner_size-1),我们可以通过使得的填充padding=kerner_size-1来实现图像大小不进行变化。

   <img src="C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20230106135920815.png" alt="image-20230106135920815" style="zoom:33%;" />

3. 


$$
H_{out} = \lfloor \frac{H_{in} - H_k + 2P}{s} \rfloor + 1
$$

$$
W_{out} = \lfloor \frac{W_{in} - W_k + 2P}{s} \rfloor + 1
$$




***

***

##### **DCNN**：

在 DCNN(Deep Convolutional Neural Network)  中，网络的前面部分通常包含多个卷积层，用于提取图像中的特征。后面部分可以使用全连接层（也称为密集层）或者另一个卷积层来进行预测或分类。

***

**DCNN的张量维度变化**

在MNIST的训练中，LeNet本质上是一种DCNN的形式，在卷积层输出后进入到全连接层（好像也有人叫密集层?）

从卷积层输出的张量以图像的四维张量输出，形状为 (batch_size, channels, height, width)，而全连接层的输入是一个 2 维张量，形状为 (batch_size, features)。在这个过程中，通常需要使用 view 函数对卷积层的输出做一次 reshape 操作，使其符合全连接层的输入要求。

```
 x = x.view(x.size(0), -1)
```

- 将卷积网络出来的结果展开到x的第一维度x.size(0)，即bitch_size，将卷积层的输出整理成一维向量，方便输入到全连接层。
- 在这里，view 函数的第一个参数是 batch_size，第二个参数是 -1，意思是将 channels * height * width 的值作为 features 的值。假设 x 是一个 4x3x2 的张量（灰度图默认通道数为1），那么 view 函数可以把 x 转换成一个 12x4 的张量，其中 12 是由 3x2 计算得到的，4 是 x 的第一维的大小。



***

#### 激活函数F.log_softmax

**多分类问题softmax**

Softmax是指数标准化函数，又称为归一化指数函数，将多个神经元的输出，映射到 (0,1) 范围内，并且归一化保证和为1，从而使得多分类的概率之和也刚好为1。其公式如下：
$$
S_{i}=\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}} \
$$

$$
f(c) = {\exp(c) \over \sum_{d} \exp(c_d)}.
$$
**log_softmax**

F.log_softmax是pytorch中的函数，它可以对一个张量进行log softmax归一化。log softmax是一种常用的归一化方法，其中归一化的结果是在对张量每一维进行softmax之后再对每一维取log。这种归一化方法常用于多分类问题，因为它可以将每一维的输出映射到概率分布上，使得每一维的输出都在0到1之间。使用log_softmax的原因是为了避免输出概率过小而导致的数值稳定性问题。

使用F.log_softmax的时候，需要注意的是它的输入张量的维度必须是2维，即(batch_size, num_classes)。对于输出张量，它的维度也是2维的，即(batch_size, num_classes)，其中第二维的长度等于输入张量的第二维的长度。
$$
L_{i}=log\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}}
$$

```
  out = F.log_softmax(x, dim=1)
```

F.log_softmax是PyTorch中的一个函数，用于对一个Tensor进行log_softmax操作。其中，dim参数用于指定log_softmax操作的维度。例如，如果dim=1，则log_softmax操作将在第一维上进行，即对每一行元素进行log_softmax操作。如果dim=0，则log_softmax操作将在第零维上进行，即对每一列元素进行log_softmax操作。

在实际的运算中是按下面的式子执行的，在加快运算速度的同时，保证数据的稳定性。
$$
L_{i}=log\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}}=log\frac{e^{z_i}/e^{M}}{\sum_{j}^{K}{e^{z^j/e^{M}}}}=log\frac{e^{(z_i-M)}}{\sum_{j}^{K}{e^{(z^j-M)}}}=(z_i-M)-log(\sum_{j}^{K}{e^{(z^j-M)}})
$$

```
# 定义卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 样本输入数据为一个batch_size大小的28*28灰度图
        # nn.conv2d(in_channels,out_channels,kernel_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=5,
                stride=1,
                padding=2   # 使得图片大小不变化仍然为28*28
            ),
            # output.shape = (20,28,28)
            nn.ReLU(),
            # 当不指定最大池化层的步长的时候默认为stride=kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2)
            # output.shape = (20,14,14)
        )

        # input.shape=(20,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=5,
                stride=1,
                padding=2   # 使得图片大小不变化仍然为14*14
            ),
            # output.shape = (50,14,14)
            nn.ReLU(),
            # 当不指定最大池化层的步长的时候默认为stride=kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2)
            # output.shape = (50,7,7)
        )
        # 全连接层
        self.fc1 = nn.Linear(50*7*7, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # 输入的x为bitch_size*1*28*28的张量
        # 卷积层和池化层，激活函数都已经利用Sequential封装在conv1和conv2里面了
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积网络出来的结果展开到x的第一维度x.size(0)，即bitch_size，讲卷积层的输出整理成一维向量，方便输入到全
        # 连接层中。x表示要调整形状的张量，shape是一个元组，表示新的形状。当shape中的某一维设置为-1时，
        # 表示自动计算该维的大小，使得张量的总元素数不变。
        # 展开成（batch_size,50*7*7)
        x = x.view(x.size(0), -1)
        # 进入全连接层1
        x = self.fc1(x)
        x = F.relu(x)   # batch*50*7*7->batch*500
        # 进入全连接层2
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)   # batch*500->batch*10
        return out
```



***

#### 封装训练函数train()

将整个模型的正向传播，梯度下降，优化器进行封装到一个函数train()中

##### model.train（）

model.train是pytorch中模型的一个方法，用于将模型设置为训练模式。在训练模式中可以启动一些特殊的功能，例如自动调整自己的参数（权重和偏置)

##### optimizer.zero_grad()

optimizer为前面指定好的优化器

而optimizer.zero_grad()是pytorch中优化器的一个方法，它将模型中所有参数的梯度清零。这么做是因为在计算参数的梯度时，pytorch会累加之前所有梯度的值。所以，在每次迭代之前，我们需要调用optimizer.zero_grad()来将所有参数的梯度清零。这样可以避免梯度累加对当前梯度计算的影响。

一般在训练循环的开头调用optimizer.zero_grad()，这样在计算一个batch的梯度时，梯度就不会被累加。例如：

```
for input, label in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```

##### 损失函数F.nll_loss()负对数似然损失函数

F.nll_loss是PyTorch中的负对数似然损失函数（Negative Log Likelihood Loss），它可以用于多分类任务（比如图像分类）中。它的计算方法是对每个类别计算交叉熵损失，然后将结果求和。

注意：在使用F.nll_loss之前，模型输出的概率值通常需要先进行归一化，一般和F.log_softmax搭配使用

`loss = F.nll_loss(output, target)`

其中，output是模型输出的概率值，是一个形状为(batch_size, num_classes)的张量；target是真实标签，是一个形状为(batch_size)的张量，每个元素对应每个样本的真实标签。
$$
 L = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i \mid x_i) 
$$


##### 触发反向传播loss.backward

##### 优化器更新参数optimizer.step()

##### 打印训练信息

```
 def train(model, device ,optimizer, epoch, train_loader):
    # model.train()是PyTorch中模型的一个方法，用于将模型设置为训练模式。
    # 在训练模型的时候，我们通常会将模型设置为训练模式，以启用一些特定的功能，例如自动求导、批标准化等。
    model.train()
    # enumerate()函数会将数据加载器的每个元素（即一批训练数据）拆分成两部分，一部分是索引值（即批次编号），一部分是元素本身（即一批训练数据）。
    # 我们使用了两个变量batch_idx和(data, target)分别存储索引值和元素本身
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 在每次训练一个新的epoch的时候需要清空优化器内部的梯度参数，否则优化器内部梯度会累加
        optimizer.zero_grad()
        output = model(data)
        # 选择NLLLoss作为损失函数,输入为表示这个样本属于10个类别的概率和标签
        loss = F.nll_loss(output, target)
        losss.append(loss)
        # 计算损失值，触发方向传播，计算出模型中所有可学习参数的梯度
        loss.backward()
        # 调用Adam优化器的step方法更新模型的参数
        optimizer.step()
        # 打印训练信息
        # 当前的训练轮数epoch，当前的索引批次batch_idx，当前批次的大小len(data),当前训练集的大小len(train_loader)
        # 和当前损失值loss.item()
        if(batch_idx+1) % 30 == 0:
            print('Train epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,
                                                                       batch_idx*len(data),
                                                                       len(train_loader.dataset),
                                                                       100*batch_idx/len(train_loader),
                                                                       loss.item()
                                                                       )
                  )
```

***

#### 封装测试函数test()

##### 评估模式的进入

模型评估是指在训练完模型后，使用测试数据来对模型的泛化能力进行评估的过程。在 PyTorch 中，可以使用 model.eval() 方法将模型设置为评估模式。评估模式与训练模式有一些区别，比如不会计算梯度、不会更新权重等。通常，在评估模型的时候需要**禁用随机化**，以便于比较结果的准确性。

```
model.eval()  指定模型进入测试模式，不再计算梯度
```

```
 with torch.no_grad():   # 不计算梯度
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
```

**评估指标loss和accuracy**

- losses：将每一个batch中的损失值求和得到总样本数的损失值。

- accuracy_number会比较预测结果和真实值中正确样本的数量

  `correct_number = pred.eq(target.view_as(pred)).sum().item()`

  - `view_as(pred)` 是 PyTorch 中的一个张量操作，它将调用者的形状和给定张量 `pred` 的形状相同。这意味着，如果调用者是一个 2 维张量，并且 `pred` 是一个 3 维张量，那么在调用 `view_as(pred)` 后，调用者的形状也将变为 3 维。这一操作可以方便地将两个张量的形状匹配起来，以便进行数学操作（如加法，乘法等）。

  - `.eq`是 PyTorch 中的一个函数，它的作用是比较两个张量的每个元素是否相等。如果两个元素相等，就返回1，否则返回0。比如：

    ```
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 4])
    c = a.eq(b)
    print(c)  # tensor([1, 1, 0])
    ```

  - `.sum（）`函数用于求张量里面所有元素的和，即返回正确预测数量的一个一维张量

  - `.item()` 是 PyTorch 中的一个方法，它可以**将一个标量张量中的数据**从张量的形式转换为 Python 对象的形式，并返回这个数据。所以我们使用 `.item()` 来将这个数据从张量的形式转换为数字的形式，这样才能将它赋值给 `correct_number` 这个变量。

```
# 封装准确率函数
def accuracy_number(output,target):
    # output的维度为bitch_size*10
    # torch中的max函数会返回一个包括输入张量的最大值和最大值的索引的元组，利用[1]来获得最大值的索引
    # keepdim使得保留output的最大维度
    pred = output.max(1,keepdim=True)[1]
    correct_number = pred.eq(target.view_as(pred)).sum().item()
    return correct_number
```

```

def test(model, device, test_loader):
    model.eval()    # 将模型设为评估模式
    test_loss = 0
    correct_number = 0
    # torch.no_grad上下文管理器可以在不计算梯度的情况下进行预测
    with torch.no_grad():   # 不计算梯度
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target)
            correct_number += accuracy_number(output, target)
    # test_loader.dataset保留了输入数据的原始信息
    test_loss = test_loss/len(test_loader.dataset)  # 计算平均损失率
    test_accuracy = correct_number/len(test_loader.dataset)
    accuracys.append(test_accuracy)
    print('\nTest loss:{:.4f},\t Accuracy：{}'.format(test_loss,test_accuracy))
    print('-------批次分割线---------')
```



***



```
from torch.utils.data import DataLoader

# 定义数据集
dataset = MyDataset()

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 迭代加载数据
for data in dataloader:
    # 处理数据
    process(data)
```



```

```

***

**卷积层Conv1d和Conv2d**

`nn.Conv2d` 和 `nn.Conv1d` 都是 PyTorch 模块，它们都实现了卷积层，但是它们在执行卷积的维数上有所不同。

`nn.Conv2d` 是一个 2D 卷积层，它在两个空间维度上执行卷积，通常是图像或特征图的高度和宽度。它通常用于图像分类和对象检测等任务。

`nn.Conv1d` 是一个 1D 卷积层，它在单个维度上执行卷积，通常是序列的时间维度。它通常用于自然语言处理和音频处理等任务。



***

**卷积层参数的选择**

选择卷积层的通道数是一个技术决策，可以帮助你构建更有效的模型。

一般来说，选择的通道数越多，模型的表示能力就越强。但是，如果通道数过多，模型也可能会过拟合。所以，选择卷积层的通道数时，需要考虑模型的泛化能力和计算复杂度。

具体来说，你可以考虑以下几点：

- 输入数据的维度：如果输入数据是高分辨率图像，那么你可能需要使用更多的通道来捕捉特征
- 模型的复杂度：如果你的模型已经很复杂了，那么使用更多的通道可能会使模型过拟合。
- 计算复杂度：卷积层的计算复杂度与通道数成正比。如果你的模型计算复杂度过高，那么使用更少的通道可能会有帮助。
- 可用的资源：如果你的模型需要在资源有限的设备上运行，那么使用更少的通道可能会更加高效。

总的来说，选择卷积层的通道数是一个权衡的过程。你需要在模型的泛化能力，复杂度和计算复杂度之间进行权衡。

***

####  **Softmax和log_softmax**

***

- Softmax是指数标准化函数，又称为归一化指数函数，将多个神经元的输出，映射到 (0,1) 范围内，并且归一化保证和为1，从而使得多分类的概率之和也刚好为1。其公式如下：
  $$
  S_{i}=\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}} 
  $$
  通俗理解，softmax函数的结果代表了类别分布，也就是说K个不同可能结果的概率分布。所以softmax经常用于深度学习和机器学习的分类任务中。
  但是Softmax会存在上溢出和下溢出的情况，这是因为Softmax会进行指数操作，当上一层的输出，也就是Softmax的输入比较大的时候，可能会产生上溢出，超出float的能表示范围；同理，当输入为负值且绝对值比较大的时候，分子分母会极小，接近0，从而导致下溢出。这时候log_Softmax能够很好的解决溢出问题，且可以加快运算速度，提升数据稳定性。

- log_Softmax其实就是对Softmax取对数，数学表达式如下所示
  $$
  L_{i}=log\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}}
  $$
  尽管，数学上log_Softmax是对Softmax取对数，但是，实际操作中是通过下面的式子来实现的
  $$
  L_{i}=log\frac{e^{z_i}}{\sum_{j}^{K}{e^{z^j}}}=log\frac{e^{z_i}/e^{M}}{\sum_{j}^{K}{e^{z^j/e^{M}}}}=log\frac{e^{(z_i-M)}}{\sum_{j}^{K}{e^{(z^j-M)}}}=(z_i-M)-log(\sum_{j}^{K}{e^{(z^j-M)}})
  $$
  **pytorch中的激活函数softmax的使用**

  F.log_softmax是pytorch中的一种激活函数，它的作用是对输入的张量进行log softmax变换。

  ```
  torch.nn.functional.log_softmax(input, dim=None, dtype=None, out=None) 
  ```

  - `input`：待变换的张量
  - `dim`：沿着哪个维度进行变换，默认是最后一个维度
  - `dtype`：输出张量的数据类型，默认与输入张量相同
  - `out`：输出张量，默认为None
  - 注意：log_softmax函数的输入应该是没有经过任何激活函数的原始输出，而不是经过softmax函数的输出。如果你想对经过softmax函数的输出做log处理，应该使用torch.log函数。





***

**最终版本**

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 从datasets下载数据集MNIST数据集包含手写数字图像，这些图像都是灰度图像，大小为28x28像素。
# 这些图像表示数字0到9中的一个数字。每个图像都被标记为属于某个特定的数字，所以MNIST数据集也被用作分类任务




BATCH_SIZE=216
EPOCHS = 10
# 使用GPU用来计算，可以通过指定CUDA：0指定第一块GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 0.0005
"""
加载MNIST数据集
"""

# pytorch里面dataset已经有MNIST的数据集
# dataset类型的函数有以下参数：
# root:表示下载数据集保存的路径，训练集和测试集都会下载到这个文件夹内部
# train：布尔类型，表示是否下载训练集
# transform；表示如何对数据转换,这里使用的transform.ToTensor函数可以使得图像转换成张量
# download:布尔类型
train_dataset = torchvision.datasets.MNIST(
    root= './dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)


"""
批量加载数据
"""
# train_loader 和 test_loader 是 PyTorch 的 DataLoader 类的实例。
# 它们提供了从数据集中按批次加载数据的功能
# 批训练数据通常是以元组(data，target)的形式保存的，其中包含了训练数据的输入和标签。
# 例如，在图像分类任务中，data是图像的张量，而target是图像的类别标签。

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size =BATCH_SIZE,
    shuffle=True
)
# 测试集的shuffle=False选择不打乱，保留数据集的顺序
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size =BATCH_SIZE,
    shuffle=False
)

# 定义卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 样本输入数据为一个batch_size大小的28*28灰度图
        # nn.conv2d(in_channels,out_channels,kernel_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=5,
                stride=1,
                padding=2   # 使得图片大小不变化仍然为28*28
            ),
            # output.shape = (20,28,28)
            nn.ReLU(),
            # 当不指定最大池化层的步长的时候默认为stride=kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2)
            # output.shape = (20,14,14)
        )

        # input.shape=(20,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=5,
                stride=1,
                padding=2   # 使得图片大小不变化仍然为14*14
            ),
            # output.shape = (50,14,14)
            nn.ReLU(),
            # 当不指定最大池化层的步长的时候默认为stride=kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2)
            # output.shape = (50,7,7)
        )
        # 全连接层
        self.fc1 = nn.Linear(50*7*7, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # 输入的x为bitch_size*1*28*28的张量
        # 卷积层和池化层，激活函数都已经利用Sequential封装在conv1和conv2里面了
        x = self.conv1(x)
        x = self.conv2(x)
        # 将卷积网络出来的结果展开到x的第一维度x.size(0)，即bitch_size，讲卷积层的输出整理成一维向量，方便输入到全
        # 连接层中。x表示要调整形状的张量，shape是一个元组，表示新的形状。当shape中的某一维设置为-1时，
        # 表示自动计算该维的大小，使得张量的总元素数不变。
        # 展开成（batch_size,50*7*7)
        x = x.view(x.size(0), -1)
        # 进入全连接层1
        x = self.fc1(x)
        x = F.relu(x)   # batch*50*7*7->batch*500
        # 进入全连接层2
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)   # batch*500->batch*10
        return out

model = ConvNet()
model.to(device)
# Adam基于传统的梯度下降算法基础上，引入了动量项和加权平均项来加速优化
optimizer = optim.Adam(model.parameters(),lr=LR)

# 定义一个封装函数train
# 该函数接受五个参数：
# model：表示要训练的模型，是一个PyTorch的模型对象。
# device：表示要使用的设备，是一个PyTorch的设备对象。
# train_loader：表示训练数据的加载器，是一个PyTorch的数据加载器对象。
# optimizer：表示优化器，是一个PyTorch的优化器对象。
# epoch：表示当前是第几个epoch，是一个整数

def train(model, device ,optimizer, epoch, train_loader):
    # model.train()是PyTorch中模型的一个方法，用于将模型设置为训练模式。
    # 在训练模型的时候，我们通常会将模型设置为训练模式，以启用一些特定的功能，例如自动求导、批标准化等。
    model.train()
    # enumerate()函数会将数据加载器的每个元素（即一批训练数据）拆分成两部分，一部分是索引值（即批次编号），一部分是元素本身（即一批训练数据）。
    # 我们使用了两个变量batch_idx和(data, target)分别存储索引值和元素本身
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 在每次训练一个新的epoch的时候需要清空优化器内部的梯度参数，否则优化器内部梯度会累加
        optimizer.zero_grad()
        output = model(data)
        # 选择NLLLoss作为损失函数,输入为表示这个样本属于10个类别的概率和标签
        loss = F.nll_loss(output, target)
        losss.append(loss)
        # 计算损失值，触发方向传播，计算出模型中所有可学习参数的梯度
        loss.backward()
        # 调用Adam优化器的step方法更新模型的参数
        optimizer.step()
        # 打印训练信息
        # 当前的训练轮数epoch，当前的索引批次batch_idx，当前批次的大小len(data),当前训练集的大小len(train_loader)
        # 和当前损失值loss.item()
        if(batch_idx+1) % 30 == 0:
            print('Train epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,
                                                                       batch_idx*len(data),
                                                                       len(train_loader.dataset),
                                                                       100*batch_idx/len(train_loader),
                                                                       loss.item()
                                                                       )
                  )
        # 损失函数的可视化


# 封装准确率函数
def accuracy_number(output,target):
    # output的维度为bitch_size*10
    # torch中的max函数会返回一个包括输入张量的最大值和最大值的索引的元组，利用[1]来获得最大值的索引
    # keepdim使得保留output的最大维度
    pred = output.max(1,keepdim=True)[1]
    correct_number = pred.eq(target.view_as(pred)).sum().item()
    return correct_number

# 封装测试函数
def test(model, device, test_loader):
    model.eval()    # 将模型设为评估模式
    test_loss = 0
    correct_number = 0
    # torch.no_grad上下文管理器可以在不计算梯度的情况下进行预测
    with torch.no_grad():   # 不计算梯度
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target)
            correct_number += accuracy_number(output, target)
    # test_loader.dataset保留了输入数据的原始信息
    test_loss = test_loss/len(test_loader.dataset)  # 计算平均损失率
    test_accuracy = correct_number/len(test_loader.dataset)
    accuracys.append(test_accuracy)
    print('\nTest loss:{:.4f},\t Accuracy：{}'.format(test_loss,test_accuracy))
    print('-------批次分割线---------')
# 正式部分开始训练
losss = []
accuracys=[]
for epoch in range(1,EPOCHS+1):
    train(model,device,optimizer,epoch,train_loader)
    test(model, device, test_loader)

# 注意这里的张量还在GPU上面运行，GPU上面的张量无法使用numpy操作，会报以下错误
#  can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# .numpy() 是一个 PyTorch Tensor 对象的方法，用于将 Tensor 转换为 NumPy 数组
# plt的操作对象为数组

losses_cpu = torch.tensor(losss, device="cpu")
accuracys = torch.tensor(accuracys, device="cpu")

plt.plot(losses_cpu)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.show()

plt.plot(accuracys)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.show()

torch.save(model.state_dict(), 'model.pt')
```

<img src="C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20230112213438623.png" alt="image-20230112213438623" style="zoom:50%;" />

<img src="C:\Users\11984\AppData\Roaming\Typora\typora-user-images\image-20230112213458709.png" alt="image-20230112213458709" style="zoom:50%;" />





***

**参考类似实现**

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
```



```
# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor()
)

# 创建数据加载器
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)
```

```
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data

EPOCH=1#训练整批数据多少次
BATCH_SIZE=64#每次批数的数据量
LR=0.001#学习率，学习率的设置直接影响着神经网络的训练效果

train_data=torchvision.datasets.MNIST(#训练数据
     root= './mnist_data/',
     train=True,
     transform=torchvision.transforms.ToTensor(),
     download=True
     )
test_data = torchvision.datasets.MNIST(
     root='./mnist_data/', 
     train=False,
     transform=torchvision.transforms.ToTensor(),
     download=True
     )
# 批量加载
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# 数据可视化
images, label = next(iter(train_loader))
images_example = torchvision.utils.make_grid(images)
images_example = images_example.numpy().transpose(1,2,0) # 将图像的通道值置换到最后的维度，符合图像的格式
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
images_example = images_example * std + mean
plt.imshow(images_example )
plt.show()

image_array,_=train_data[0]#把一个批数的训练数据的第一个取出
image_array=image_array.reshape(28,28) #转换成28*28的矩阵
plt.imshow(image_array)
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height输入
                out_channels=16,    # n_filters输出
                kernel_size=5,      # filter size滤波核大小
                stride=1,           # filter movement/step步长
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=，(kernel_size-1)/2 当 stride=1填充
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN().cuda()
print(cnn)  # 显示神经网络
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # Adam优化函数
loss_func = nn.CrossEntropyLoss()   # 损失函数（损失函数分很多种，CrossEntropyLoss适用于作为多分类问题的损失函数）

# training and testing
for epoch in range(EPOCH):#训练批数
    for step, (x, y) in enumerate(train_loader):   # 每个批数的批量
        b_x = x.cuda()   # batch x
        b_y = y.cuda()   # batch y
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        print('epoch: %s   step: %s   loss: %s'%(epoch, step, loss))
        optimizer.zero_grad()           # 梯度清零
        loss.backward()                 # 损失函数的反向传导
        optimizer.step()                 # 对神经网络中的参数进行更新

#在测试集上测试，并计算准确率
print('**********************开始测试************************')

for step, (x, y) in enumerate(test_loader):
    test_x, test_y = x.cuda(), y.cuda()
    test_output = cnn(test_x)
    
    # 以下三行为pytorch标准计算准确率的方法，十分推荐，简洁明了易操作
    pred_y = torch.max(test_output.cpu(), 1)[1].numpy()
    label_y = test_y.cpu().numpy()
    accuracy = (pred_y == label_y).sum() / len(label_y)

print(test_output) #查看一下预测输出值
print('acc: ', accuracy)

```

```
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out
```







***

# 图像分割

顾名思义，图像分割就是指将图像分割成多个部分。在这个过程中，图像的每个像素点都和目标的种类相关联。图像分割方法主要可分为两种类型：语义分割和实例分割。语义分割会使用相同的类标签标注同一类目标（下图左），而在实例分割中，相似的目标也会使用不同标签进行标注

https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650786169&idx=3&sn=78daf0132ed86258bca062d69688788a&chksm=871a0d07b06d8411fe2f92abf753f6737acab1cc2119883fb1fad5b370178e2ec728968c7823&scene=21#wechat_redirect
