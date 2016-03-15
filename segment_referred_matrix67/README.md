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

我们不妨把 max_word 从 5 改为 4，最多 4 个字构成词，然后继续运行，这样的话，发现可以搜索 12 万行，再多也会超过内存限制

12 万行的结果如下

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


其实，效果看上去也比较一般
