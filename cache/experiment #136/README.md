# 实验说明

因为，今天早上发现这种,分$\rightarrow$合$\rightarrow$分的形式并不能很好的发挥作用。所以，我进一步把计算方法由原来的，
```python
para_00 = self.sigmoid(self.para_fc0_0(feat_para[:, idx, :]))
para_01 = self.sigmoid(self.para_fc0_1(feat_para[:, idx + 1, :]))
sum_para = torch.cat([para_00, para_01], 1)
para_00 = self.relu(self.para_fc0_2(sum_para)).view(b, -1, 1, 1)
para_01 = self.relu(self.para_fc0_3(sum_para)).view(b, -1, 1, 1)
```
改成了，
```python 
sum_para = torch.cat([feat_para[:, idx, :], feat_para[:, idx + 1, :]], 1)
sum_para = self.sigmoid(self.para_fc0_0(sum_para))
para_00 = self.relu(self.para_fc0_1(sum_para)).view(b, -1, 1, 1)
para_01 = self.relu(self.para_fc0_2(sum_para)).view(b, -1, 1, 1)
```
在数学公式上的区别在于，之前是，
$$
para_{00} = Sigmoid(W_{00} \times f_0)
$$
$$
para_{01} = Sigmoid(W_{01} \times f_1)
$$
$$
SumPara = (para_{00}, para_{01})
$$
$$
para_{00} = ReLU(W_{02} \times para_{00})
$$
$$
para_{01} = ReLU(W_{03} \times para_{01})
$$
现在是， 
$$
SumPara = (f_0, f_1)
$$
$$
SumPara = Sigmoid(W_{00} \times SumPara)
$$
$$
para_{00} = ReLU(W_{01} \times SumPara)
$$
$$````
para_{01} = ReLU(W_{02} \times SumPara)
$$

# 实验结果

|Epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|80.5|87.1|94.4|
|480|80.2|87.1|94.2|
|440|80.5|87.3|94.7|
|400|80.1|87.2|94.4|
|360|79.9|87.1|94.5|
|320|80.5|87.3|94.6|

## 实验情况

1. 发现，即使训练了520次，损失函数的值还在不断的下降。比如，一般情况下，交叉熵对应的损失函数应该是在`3.0`上下波动。但是，在本实验中交叉熵的损失函数只下降到了`6.0`左右。是否意味着，实验还有比较大的上涨空间呢？

2. 在我之前的实验中，采用分$\rightarrow$合$\rightarrow$分，并没有很好的按照我所想的那样进计算的。但事实上，这应该是不影响我们的计算的。毕竟都是由`合`生成的`分`向量，只是说记的号不一样而已。为什么会出现这种情况呢？不应该呀？调一下参数，或者增加一层全连接会不会好一点。

或者，从含义出发。我们再想一个更深层次的解释？ 不要为了改实验而改实验，要明确改进的方案，才是有效的。