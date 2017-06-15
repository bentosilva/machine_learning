Notes
======

这里采用了 4 种度量方式：皮尔斯相关系数、KS-test 的 p-value、余弦相似度、正规化欧几里德距离，而且都调整为越大 (接近 1) 越好的模式

经过一些测试，发现

1. KS-test 的 p-value 并不好。通常 ks-test 是针对连续分布，而 scipy.stats.ks_2samp 虽然可以比较两个离散分布，
    但是希望 “sample observations assumed to be drawn from a continuous distribution, sample sizes can be different”
    显然，我们这里并不是从连续分布来抽取样本的， 而是本质就是离散分布

2. 余弦相似度指标还可以，但是还是不如皮尔斯相关系数来的更精准

3. 欧式距离，类似余弦相似度指标，但是不如余弦指标

小结

- 首先应该使用皮尔斯相关系数，能很准确的判断待定分布的 “形状” 是否和 benford's law 分布相似，相关度多少 (两条直线，不管倾斜角度如何，相关度都是 1，因为形状相同)
- - 然后在上面的基础上，再使用 cosine 相似度 (或者欧式距离等 based on distance 的指标)，能够计算待定分布和 benford's law 分布的距离数据
- - 最开始，一定要清洗数据，去掉无效数字、年份、月份、编号等




