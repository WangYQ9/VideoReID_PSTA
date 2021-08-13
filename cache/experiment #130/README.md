# 实验修改说明

实验的基础和`#129`相同，都对`FC`的对应功能做了修改。本实验将数学公理，
$$
para_0 = ReLU(W_1 \times Sigmoid(W_2 \times f_0)) \tag{1}
$$
$$
para_1 = ReLU(W_1 \times Sigmoid(W_2 \times f_1)  \tag{2}
$$
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_0 \times f_0 +para_1 \times f_1)^2 \tag{3}
$$
等式(3)改成了
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_1 \times f_0 +para_0 \times f_1)^2 \tag{4}
$$

# 实验结果

|epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520(BE)|$\color{red}81.8$|87.0|95.1|
|520(AF)|81.6|$\color{red}87.3$|$\color{red}95.3$|
|480(BE)|81.7|86.7|95.0|
|480(AF)|81.4|87.1|95.2|
|440(AF)|81.3|$\color{red}87.3$|95.2|
|400(AF)|81.1|87.2|95.2|
|240(BE)|81.3|87.0|94.9|

## 实验分析

这个实验充分说明了，之前那样的`FC`层用法确实是存在问题的。修改过来后，性能出现了比较稳定的上调，尤其是`mAP`的指数。但上调的比例不算大啊。想要做到`Rank1-89`你还要再想更多的办法！

目前只是做到了暂时在`Rank1-87`上站住了脚。真的还不够啊！==怎么办？==

别放弃！ 想涨点的办法！还有什么没想到的？？？