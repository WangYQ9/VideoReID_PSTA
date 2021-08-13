# 实验说明

在检查代码的时候发现，我不单在第`0`阶特征向量的生成中用到了`ReLU`。在其他几阶特征向量的生成中也用到了`ReLu`。所以，既然在第`0`阶中没有用到`ReLU`那么在其他各阶中也不应该出现`ReLU`。

本次实验则是，将这几个不必要的`ReLU`删除后的模型。

## 实验结果

|epoch|mAP|rank1|rank5|
|:--:|:--:|:--:|:--:|
|520|76.6|83.4|92.8|
|480|75.6|82.7|92.0|
|440|$\color{red}78.8$|84.8|$\color{red}94.2$|
|400|78.6|$\color{red}85.1$|94.1|
|360|75.4|82.7|92.4|
|320|78.7|$\color{red}85.1$|94.1|


# 接下来的实验方向

1、尝试只使用最后一个特征向量进行训练和测试。（这个我之前好像试过，效果没有现在的好）。

2、放弃第`0`阶特征向量参与训练特征向量和测试特征向量的运算。

3、用论文[1]的方法，对合并特征向量的权重加以归一化的处理。

4、用论文[2]中使用到的度量函数，
$$
S(x_i,x_j) = \frac{2}{e^{||x_i-x_j||_2} + 1}
$$
替代余弦相似度度量合并向量的权重。


# 参考文献

[1] STA:Spatial-Temporal Attention for Large-Scale Video-based Person Re-Identification
[2] Adaptive Graph Representation Learning for Video Person Re-identification