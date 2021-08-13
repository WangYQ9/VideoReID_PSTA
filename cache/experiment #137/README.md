# 实验说明

因为，几次对分$\rightarrow$合$\rightarrow$分的实验设计都没有得到一个比较好的结果。说明，这个想法没有那么容易就能找到能`work`的办法。
但是，从`实验#129`得到的结论是，以自身为导向的`SENet`是比较起作用的。所以，在实现下列等式的时候

$$
(\alpha \times f_0 + \beta \times f_1)^2 + \gamma \times f_0 + \theta \times f_1
$$

参数$\gamma$和$\theta$是通过下列等式生成的，

$$
para_0 = Sigmoid(W_0 \times [f_0,f])
$$
$$
para_1 = Sigmoid(W_0 \times [f_1,f])
$$
$$
\theta = para_0 = ReLU(W_1 \times para_0)
$$
$$
\gamma = para_1 = ReLU(W_1 \times para_1)
$$
所以有
$$
(\alpha \times f_0 + \beta \times f_1)^2 + \gamma \times f_0 + \theta \times f_1 = (\alpha \times f_0 + \beta \times f_1)^2 + para_0 \times f_0 + para_1 \times f_1
$$
注意，这里的参数$para_0$和$para_1$由特征向量相互生成的，输入并不是自身。

但由`实验#129`的结果倒推，输入为自身可能结果会更好一些。所以，可以尝试这么做做，这应该是目前能想到的最优方案了。

# 实验结果

| Epoch | mAP  | Rank1 | Rank5 |
| :--:|:--:|:--:|:--:|
|520|82.2|87.8|95.9|
|480|82.2|87.8|$\color{red}96.0$|
|440|$\color{red}82.4$|87.8|95.9|
|400|82.3|$\color{red}88.0$|95.9|

# 实验分析 

改动了生成$\gamma$和$\theta$的生成办法，从结果上来看，对整体模型的提升效果有限。比较类似于在实验`#129`中把卷积的偏置去掉的情况。但总的来说，`mAP`和`Rank5`都得到了一定程度的提升，`Rank1`也勉强够到了`88`，总的来说，算是勉强合格吧。但是，这个结果还是拿不出来。





