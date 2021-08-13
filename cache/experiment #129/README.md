# 实验修改说明



原来计算权重向量的代码是通过
```
	feat_para = self.avg(feat_map.view(b * t, c, w, h)).view(b*t, -1)
	para = self.sigmoid(self.para_fc0_0(feat_para))
	para = self.relu(self.para_fc0_1(para))
```
因为，数据特性中`batch`和`seq_len`是两种不同类型的数据。所以，不可以将其通过`view`放在一起处理。

修改成如下的分批处理。
```
for idx in range(0, t, 2):
	para_00 = self.sigmoid(self.para_fc0_0(feat_para[:, idx, :]))
	para_00 = self.relu(self.para_fc0_1(para_00)).view(b, -1, 1, 1)
	para_01 = self.sigmoid(self.para_fc0_0(feat_para[:, idx + 1, :]))
	para_01 = self.relu(self.para_fc0_1(para_01)).view(b, -1, 1, 1)
	gap_map0 = para_00 * feat_map[:, idx, :, :, :] + para_01 * feat_map[:, idx + 1, :, :, :]
	gap_map0 = gap_map ** 2
	gap_feat_map0.append(gap_map0)
```

数学原理和之前没修改前的是一样的， 
$$
para_0 = ReLU(W_1 \times Sigmoid(W_2 \times f_0)) \tag{1}
$$
$$
para_1 = ReLU(W_1 \times Sigmoid(W_2 \times f_1)  \tag{2}
$$
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_0 \times f_0 +para_1 \times f_1)^2 \tag{3}
$$


# 实验结果

|epoch|mAP|rank1|rank5|
|:--:|:--:|:--:|:--:|
|520|81.9|87.8|95.3|
|480|$\color{red}82.2$|$\color{red}88.2$|$\color{red}95.4$|
|440|81.7|87.7|95.2|
|400|81.6|87.8|95.2|
|360|$\color{red}82.0$|$\color{red}88.3$|$\color{red}95.4$|
|320|81.5|88.0|95.0|

## 实验分析

别着急，这个结果还不会是最后的结果！`Rank1-88.3%`还不够看，不过最起码有了希望！晚点作分析。

