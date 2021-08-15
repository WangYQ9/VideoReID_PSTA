# 实验说明

在实验`#137`中的做法是，

$$
para_0 = Sigmoid(W_0 \times [f_0,f])\tag{1}
$$
$$
para_1 = Sigmoid(W_0 \times [f_1,f])\tag{2}
$$
$$
\theta = para_0 = ReLU(W_1 \times para_0)
$$
$$
\gamma = para_1 = ReLU(W_1 \times para_1)
$$
$$
(\alpha \times f_0 + \beta \times f_1)^2 + \gamma \times f_0 + \theta \times f_1 = (\alpha \times f_0 + \beta \times f_1)^2 + para_0 \times f_0 + para_1 \times f_1
$$

现在的做法是，把公式$(1)$和$(2)$中的$[f_0, f]$和$[f_1, f]$改成$f_0$和$f_1$。这样就完全和实验`#129`吻合了。

# 试验结果

|Epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|
