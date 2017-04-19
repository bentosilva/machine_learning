基于 matrix67 的无训练数据分词文章 http://www.matrix67.com/blog/archives/5044

初试
=====

使用 icwb 数据做测试，如下
```
$ python2.7 segment_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_result
Avg len:  3.77535585651
Avg freq:  1.1521831998e-05
Avg left ent:  0.0816796763844
Avg right ent:  0.0810123828582
Avg aggreg:  542.496369401
```

默认的阈值是 max_word=5, min_freq=0.00005, min_entropy=2.0, min_aggreg=50
```
$ cd data/
$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_result
......
=== TOTAL TRUE WORDS RECALL:    0.637
=== TOTAL TEST WORDS PRECISION: 0.452
=== F MEASURE:  0.529
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.093
=== IV Recall Rate:     0.671
```

结果不是很好，说明默认的参数并不好，不适合我们这个测试文档


调整 1.
========

根据之前的统计结果，看到左右邻居熵的均值在 0.081 多一些，并不高，我们这里设置了 2.0，过高了，改为 0.1 试试看

聚合达到了 542.5，而我们默认参数只有 50，过小了，改为 600 看看

由于我们已经生成了统计结果文件 candidates_statistics.csv，故此这里直接使用新的参数值来过滤并更新 good_words.csv 文件即可

```
$ python2.7 segment_re_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_result_entropy0.1_aggreg600 5 0.00005 0.1 600
```

都不需要跑评估程序，只要看看 good_words.csv 中只有 281 个词，显然条件过于苛刻了

回头看，使用默认参数时，得到的 good_words.csv 共计 672 个词，其实也不多啊，难道是默认的条件也苛刻？

比较一下默认参数和上面的参数，直觉上，熵改为 0.08 看看，已经比默认的 2.0 宽松多了； 聚合的值设为 600 看来是太大了，改为 300 试试看

563 个好词，仍然不够，继续调整，继续减少聚合值，改为 100 看看 ==> 1183 个好词，似乎好了不少，评估以下

```
$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_result_entropy0.08_aggreg100
=== TOTAL TRUE WORDS RECALL:    0.633
=== TOTAL TEST WORDS PRECISION: 0.456
=== F MEASURE:  0.530
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.105
=== IV Recall Rate:     0.665
```

和默认参数结果差距不大

考虑了一下，应该是因为测试样本太少的缘故，导致词频不够 (凝固需要计算词频，词频过少会导致平均熵过大，因为所有词的出现频次都少)

理论上，应该使用大量样本来学习，学到的 good_words.csv 再来对测试样本进行分词；而不是直接使用测试样本来学习


调整 2.
=======

使用一些微博语料来训练，然后再使用训练得到的 good_words.csv 来做分词

这样，需要改一下 segment_runner.py，因为需要区分训练文件和待分词文件，这两个不再是同一个文件了

另外，微博语料过大，我测试机器内存不够，由于微博中有很多的非中文，那么实现一个 filter_chinese.py 先清理微博语料文件 

微博语料只取 10000 行，否则机器内存跑不动，算法内存占用太大了；先清理，再训练、分词，再评估，如下：

```
清理语料，减少文件容量
$ python2.7 filter_chinese.py ./data/yuebingwb ./data/yuebingwb_filtered

使用默认参数进行训练和分词
$ python2.7 segment_runner.py ./data/yuebingwb_filtered ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default
calculating aggregations ....
Avg len:  3.80394048974
Avg freq:  4.59949599138e-06
Avg left ent:  0.0739596553054
Avg right ent:  0.0735969001754
Avg aggreg:  1274.05676568

$ cd data
$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_yuebing_default
=== TOTAL TRUE WORDS RECALL:    0.552
=== TOTAL TEST WORDS PRECISION: 0.359
=== F MEASURE:  0.435
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.070
=== IV Recall Rate:     0.581
```

看到，由于语料仍然过小，good_words.csv 只有 500多，结果还不如直接拿待分词文件来训练得到的结果好。

想想也是自然的，因为其他语料如果小的话，很可能不包含待分词文件中的词啊

如果想有突破的话，要么使用大内存机器训练，要么修改算法，不要一次性算出所有 candidate 组合，lazy 化，或者 MapReduce

看到其实 candidates_statistics.csv 文件还是很大的，那么是不是可以通过调整参数来得到更多的 good_words.csv ，从而得到好结果呢？

rerun 一次比较麻烦，为了更快的测试不同参数下得到的 good words，不妨使用类似下面的 awk 语句，直接调整参数

$ awk -F'\t' '$2+0>0.00001&&$3+0>0.07&&$4+0>0.07&&$5+0>1{print}' candidates_statistics.csv > good_words.csv

但是问题仍然是语料不足，调整几次参数并观察得到的结果文件，发现 candidates 的数据并没有明显的区分度，无法正确区分好词坏词


调整 3.
=========

修改 matrix67_segment.py 实现按 batch 训练，从训练文件中连续读出一些行，超过 batch_size 后，就对这批文本进行训练

避免了一次性 load 过大的文件，导致内存超出系统限制

```
$ python2.7 segment_runner.py ./data/yuebingwb ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default
Avg len:  3.80378937044
Avg freq:  4.59740929117e-06
Avg left ent:  0.0739574155469
Avg right ent:  0.0735934635117
Avg aggreg:  1276.71891925
```
看到，和前面的结果完全一致，说明逻辑没有问题，那么来跑原始的超大的微博数据

这个文件有 62 万行之多，即使前面的改动也无法都跑完，需要进一步把结果空间压榨

我们不妨把 max_word 从 5 改为 4，最多 4 个字构成词，然后继续运行，这样的话，发现可以搜索 11 万行，再多也会超过内存限制

11 万行的结果如下

```
$ python2.7 segment_runner.py ./data/yuebingwb_origin ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default
Avg len:  3.31473730036
Avg freq:  2.04383795437e-06
Avg left ent:  0.099274889243
Avg right ent:  0.0985770575878
Avg aggreg:  1857.47820923

$ wc -l good_words.csv
548

$ wc -l candidates_statistics.csv
1957102
```

不用评估，也知道 good words 太少了；

使用上面的 awk 技巧，选用不同的参数选取 good words

```
$ python2.7 segment_re_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default 4 0.00001 0.07 1

$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_yuebing_default
=== TOTAL TRUE WORDS RECALL:    0.629
=== TOTAL TEST WORDS PRECISION: 0.501
=== F MEASURE:  0.558
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.084
=== IV Recall Rate:     0.662
```

结果有所提高，但是效果并不好


调整 4.
========

前面的尝试中，candidates_statistics.csv 中是包含单个字的，只是 good_words.csv 中会过滤掉单个字

训练脚本运行的统计数据也是把单个字计算在内的统计结果，而单个字的凝固都是 0，而且他们不会被最终收录 good_words.csv

这次，我们仍然计算单个字的频率、邻居熵，但是统计结果时，不把单个字纳入统计，以在统计结果时消除单字的影响

单字也不要导出到 candidates_statistics.csv 文件中

修改 matrix67_segment.py 并运行

```
$ python2.7 segment_runner.py ./data/yuebingwb_origin ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default
Avg len:  3.32220903678
Avg freq:  1.53782635793e-06
Avg left ent:  0.0944361883832
Avg right ent:  0.0937640397724
Avg aggreg:  1863.47395985
```

查看 candidates_statistics.csv 文件，并使用前面提到的 awk 技巧，确定参数

$ python2.7 segment_re_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_yuebing_default 4 0.00005 1 7

$ cd data
$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_yuebing_default
=== TOTAL TRUE WORDS RECALL:    0.682
=== TOTAL TEST WORDS PRECISION: 0.515
=== F MEASURE:  0.587
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.078
=== IV Recall Rate:     0.719

比之前结果好的有限


经过我的分析，我觉得利用这个方法从头直接分词并不靠谱

但是如果有词库的前提下，做新词自动发现，也就是选取指标高的且不在词库中的词，会比较好



jieba 分词
============

新词发现模式

$ python2.7 jieba_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_jieba_new new

$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_jieba_new

=== TOTAL TRUE WORDS RECALL:    0.787
=== TOTAL TEST WORDS PRECISION: 0.853
=== F MEASURE:  0.818
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.583
=== IV Recall Rate:     0.799


精确模式

$ python2.7 jieba_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_jieba_precise precise

$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_jieba_precise

=== TOTAL TRUE WORDS RECALL:    0.787
=== TOTAL TEST WORDS PRECISION: 0.853
=== F MEASURE:  0.818
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.583
=== IV Recall Rate:     0.799


其实，jieba 分词的效果看上去也比较一般



调整 5.
==============

前面看到，其实这个方法并不适合直接用来分词，而是用于发现词库中没有包含的未登录新词

故此，我们看看是否可以通过发现的新词，来扩充 jieba，让 jieba 分词的效果更加准确

那么，我们做了如下的调整

1. 聚合度的计算结果，在原来的基础上取一次 log，信息论中表示两个事件的互信息 PMI

使用 icwb 的 pku 语料进行训练，然后先不做分词测试，直接先把所有非单字的词整理到 candidates_statistics.csv 文件中
```
ws = Words('./data/pku_test.utf8')
ws.train()
```
wc -l candidates_statistics.csv
431089 candidates_statistics.csv

使用下面的方法，可以把 csv 转为 gbk 并用 excel 打开，不过最终没有使用 excel 
> sed 's/\t/,/g' candidates_statistics.csv |iconv -f utf-8 -t gbk -c > test.csv


2. 实现脚本 candidate_analysis.py 来分析得到 candidates_statistics.csv 文件，得到结果如下

```
number of whole candidate terms: 431089
---------------------------- freq -----------------------------
min: 6.64390023519e-06
max: 0.00306283800842
diff: 0.00305619410818
median: 6.64390023519e-06
mean: 9.27867229504e-06
std: 2.06378030005e-05
var: 4.25918912689e-10
corrcoef: 1.0
histogram: (array([430219,    600,    141,     64,     26,     13,      5,      4,
            1,      4,      4,      1,      2,      2,      0,      1,
            0,      1,      0,      1]), array([  6.64390024e-06,   1.59453606e-04,   3.12263311e-04,
            4.65073016e-04,   6.17882722e-04,   7.70692427e-04,
            9.23502133e-04,   1.07631184e-03,   1.22912154e-03,
            1.38193125e-03,   1.53474095e-03,   1.68755066e-03,
            1.84036037e-03,   1.99317007e-03,   2.14597978e-03,
            2.29878948e-03,   2.45159919e-03,   2.60440889e-03,
            2.75721860e-03,   2.91002830e-03,   3.06283801e-03]))
---------------------------- left -----------------------------
min: 0.0
max: 4.61064359652
diff: 4.61064359652
median: 0.0
mean: 0.0715200189137
std: 0.294334503052
var: 0.0866327996867
corrcoef: 1.0
histogram: (array([400082,    368,   3482,  15304,   5046,    932,   2512,    848,
            694,    653,    360,    272,    196,    141,     96,     53,
            32,      9,      5,      4]), array([ 0.        ,  0.23053218,  0.46106436,  0.69159654,  0.92212872,
            1.1526609 ,  1.38319308,  1.61372526,  1.84425744,  2.07478962,
            2.3053218 ,  2.53585398,  2.76638616,  2.99691834,  3.22745052,
            3.4579827 ,  3.68851488,  3.91904706,  4.14957924,  4.38011142,
            4.6106436 ]))
---------------------------- right -----------------------------
min: 0.0
max: 4.66096616707
diff: 4.66096616707
median: 0.0
mean: 0.0709891027724
std: 0.293483850899
var: 0.0861327707384
corrcoef: 1.0
histogram: (array([400379,    376,  18272,    247,   5056,   2197,   1243,    891,
            803,    480,    397,    267,    171,    121,     80,     63,
            25,     12,      8,      1]), array([ 0.        ,  0.23304831,  0.46609662,  0.69914493,  0.93219323,
            1.16524154,  1.39828985,  1.63133816,  1.86438647,  2.09743478,
            2.33048308,  2.56353139,  2.7965797 ,  3.02962801,  3.26267632,
            3.49572463,  3.72877293,  3.96182124,  4.19486955,  4.42791786,
            4.66096617]))
---------------------------- aggreg -----------------------------
min: -5.17518470039
max: 11.9218113821
diff: 17.0969960825
median: 5.1138764384
mean: 4.81986799272
std: 1.89227938845
var: 3.58072128396
corrcoef: 1.0
histogram: (array([    1,     1,     8,    86,   783,  3546,  9752, 17437, 24627,
            31453, 50253, 72837, 92149, 78962, 33255, 10574,  3712,  1130,
            353,   170]), array([ -5.1751847 ,  -4.3203349 ,  -3.46548509,  -2.61063529,
            -1.75578548,  -0.90093568,  -0.04608588,   0.80876393,
            1.66361373,   2.51846354,   3.37331334,   4.22816314,
            5.08301295,   5.93786275,   6.79271256,   7.64756236,
            8.50241217,   9.35726197,  10.21211177,  11.06696158,  11.92181138]))
```
看到如下信息：

- 词频上看，绝大部分词，430219 个，都集中在最小的 1/20 桶里，词频小于 1.59453606e-04；然后，随着词频增大，词的数量减少
- 由于大部分词出现次数太少，故此 left entropy or right entropy 为 0 的非常多 (出现过少，导致左、右词缀太过于单一, 于是 p(x) = 1 and log(p(x)) = 0  ==>  -1 * p(x) * log(p(x)) = 0)
- 左右熵同样也是绝大部分词都集中在最小的 1/20 桶里，然后在 3/20 和 4/20 和 5/20 桶中数量比较大，两侧则数量少一些，但没有明显的趋势
- 聚合度上看，聚合度很小的词其实很少，大部分的词都集中在中部，然后两侧逐渐减少，类似于正态分布


3. 挑选合适的参数，来发现比较好的新词

算法上，我们希望找到左右熵大的，聚合度也大的，词频也多的；我们看到，左右熵的最大值也相对偏小一些，应该是语料偏小造成的

暂时取 aggreg 后 6 个桶中，也就是 aggreg > 6.79271256 的；left 和 right 熵去掉各自前 4 个桶的，也就是大于 left/right > 0.93 的

就暂时不管词频了，我们知道，其实词频也会影响上面两个指标

试试看，运行下面程序，生成 good_words.csv 新词文件
```
ws = Words('', min_freq=0, min_entropy=0.93, min_aggreg=6.79271256)
ws.train_from_candidates_file()

$ wc -l good_words.csv
455 good_words.csv          得到 455 个新词
```

翻了一下这些新词，结果还是不错的，大都是常见词呢！还找到了 孙中山、克林顿 这样的人名，还有一些常用词的组合

下面，我们扩展一下 filter 函数，不仅找到满足词频、左右熵、聚合度的词，而且要从 jieba 词库中排除掉已有的词，看看剩下多少！
```
import jieba

def filter(words):
    jieba.dt.check_initialized()
    for w in words:
        if w.text not in jieba.dt.FREQ and len(w.text) > 1 and w.aggreg > self.min_aggreg and\
            w.freq > self.min_freq and w.left > self.min_entropy and\
            w.right > self.min_entropy:
    ...........

```

再次运行，看到 good_words.csv 只剩下 58 个新词了。看看这 58 个词，主要有几种情况

- 词的组合，比较多，如：退出现役 附图片张 古银杏树 歪参谋 宋双亲王 喜庆气氛 澳门回归 南极考察站 等等
- 专有人名、地名等：夏世清 夏老汉 远东豹 白莲乡 户胡镇 袁曙宏 董烈宏 伊斯塔帕 薛成飞 胡斌 英雄三岛 姚铜梅 楞吉克 森喜朗 萨本望 刘太忠 李四娃
- 量词和名词的组合： (x)亿里亚尔(里亚尔是沙特货币) (x)例疯牛病 查获假(x) (x)轮投票； x 可能性比较多，故此左、右熵很高

其实，第三种情况也是第一种情况的特例；而第二种情况是比较好的，确实识别了 jieba 中没有的新词

那么，第一种(和第三种)情况是个问题，这里做了一些思考:

- 我们这里会对原始语料 5-gram 分词；而很多资料中，直接限制了窗口为 3 甚至为 2，故此这种词组合现象自然就没有了
- 我们的训练语料太小，导致一些被组合的词出现次数太少，进而导致聚合度异常变大；如果使用大语料，甚至某个垂直领域的大语料，效果会好很多
