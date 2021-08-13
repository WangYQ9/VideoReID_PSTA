# 实验修改说明

首先，我们是在当前结果最好的实验#129的基础上做的的。实验#129对应的数学公理是，
$$
para_0 = ReLU(W_1 \times Sigmoid(W_2 \times f_0)) \tag{1}
$$
$$
para_1 = ReLU(W_1 \times Sigmoid(W_2 \times f_1)  \tag{2}
$$
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_0 \times f_0 +para_1 \times f_1)^2 \tag{3}
$$
我们做的修改是，
$$
para_{sum} = Cat\big[Sigmoid(W_2 \times f_0), sigmoid(W_2 \times f_1)\big]
$$
$$
para_0 = ReLU(W_11 \times sum_para)
$$
$$
para_1 = ReLU(W_12 \times sum_para)
$$

$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_0 \times f_0 +para_1 \times f_1)^2 \tag{3}
$$
实际上也就是我们之前提到的分$\rightarrow$合$\rightarrow$分结构的第二版，第一版是实验`#132,#133`。

## 补充说明

和实验`#132,#133`相同，会出现特征向量被`ReLU`过滤的情况。所以，在实验过程中，我把`ReLU`去掉了。


# 实验结果

|Epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|79.6|87.1|94.9|
|480|79.7|86.9|94.9|
|440|79.6|86.7|94.8|
|400|79.5|86.8|94.9|
|360|79.7|86.2|95.2|

## 分析

感觉我们的思路是好的。但是，并不能通过网络把我们的思路完全体现出来。问题在那里呢？反思一下，你这么做的思路是什么？

思路是，因为需要把两帧对应的特征向量加起来。所以，权重向量的生成应该是一个和双方都有关的。但好像并不work！注意到，这里的`FC`分$\rightarrow$合的时候，用的是相同的，这有可能是导致出现问题的地方，处理数据应该保持前后一致。改一下试试！