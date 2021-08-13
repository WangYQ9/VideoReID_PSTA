# 实验说明

在我们的实验中期望公式，
$$
(\mathcal{A} \times f_0 + \mathcal{B} \times  f_1)^2 + \mathcal{C} \times f_0 + \mathcal{D} \times f_1
$$
中的$\mathcal{A}$和$\mathcal{B}$是可正可负的。所以，在参数生成的非线性激活函数的选择上，我用了`tanh()`替代原本的`sigmoid()`。但是，发现效果都不是很好，不晓得是什么原因。有可能是这种激活函数和`ReLU()`存在冲突。因为，这会导致特征趋向于零收敛。