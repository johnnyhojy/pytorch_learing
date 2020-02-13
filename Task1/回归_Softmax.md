# 线性回归模型

### 数据生成：
```python
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)
```

### 总体分成下面四个部分：

<img src="https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=4137943918,3894617166&fm=173&app=49&f=JPEG?w=302&h=218&s=3EAA78235146DD4D5AD581DB000080B1#pic_cener" >

##### Hpothesis：模型
```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)
```

##### parameters:参数
```python
from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly
```

##### Cost Function： 损失函数
```python
loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`
```

##### Goal：目标函数
```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
```

##### 模型训练：
```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

# softmax和分类模型

<img src="https://s2.ax1x.com/2019/12/29/ln6yzn.jpg#shadow" width="422" >

sigmoid 函数可以将 input 压缩到 [0,1] 的范围，但是对于分类问题来说，我们不仅要求概率范围是[0,1]，还要求所有的概率和为 1，即 ∑pi=1
为了解决此类问题，就有了 Softmax 函数，具体的函数表达式为：

<img src="https://www.zhihu.com/equation?tex=y_%7Bi%7D+%3D+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D+%5C+%5C+%5C+%5Cforall+i+%5Cin+1...C" width="211" >



### 交叉熵损失函数
交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量p与q的相似性。交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率可以被输出的误差所控制。

在特征工程中，可以用来衡量两个随机变量之间的相似度。

在语言模型中（NLP）中，由于真实的分布p是未知的，在语言模型中，模型是通过训练集得到的，交叉熵就是衡量这个模型在测试集上的正确率。

*reference*

https://www.jianshu.com/p/c02a1fbffad6

## softmax 模型
```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制
```

# 感知机

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190924153230543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM2MTE4MzY1,size_4,color_FFFFFF,t_70)