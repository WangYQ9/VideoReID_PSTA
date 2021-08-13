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
para_{sum} = Sigmoid(W_2 \times f_0) + sigmoid(W_2 \times f_1)
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
实际上也就是我们之前提到的分$\rightarrow$合$\rightarrow$分的结构。

## 备注

因为这么改动过后，会出现`ReLU`过去所有的特征表示。即，在代码块中
```
if seq_len != 1 :
	feature = self.cat_vect(gap_feat_vect)
	feature = self.relu(feature)
else :
	feature = gap_feature_vect
```
命令==`feature = self.relu(feature)`==会导致`feature`输出零向量。所以，我对应作了两个实验，一个是把`ReLU`去掉的实验，一个在去掉`ReLU`的基础上把卷积的`偏置`置零。

# 实验结果

## 实验#132

|Epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|79.9|87.3|94.9|
|480|80.0|87.5|95.0|
|440|80.1|87.3|94.9|
|400|80.0|87.3|94.8|
|360|79.9|87.3|95.1|

## 实验#133

|Epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|80.6|86.2|94.7|
|480|80.4|86.2|94.6|
|440|80.6|85.9|94.5|
|400|80.5|86.1|94.6|
|360|78.1|85.1|93.2|

# 分析

实验效果不好，应该是意料之内的。直接相加，没有办法体现任何的含义。想要不一样，必须生成有相应的权重向量。

再等等实验#134