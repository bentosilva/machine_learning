基于 matrix67 的无训练数据分词文章 http://www.matrix67.com/blog/archives/5044

初试
=====

使用 icwb 数据做测试，如下

$ python2.7 segment_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_result
Avg len:  3.77535585651
Avg freq:  1.1521831998e-05
Avg left ent:  0.0816796763844
Avg right ent:  0.0810123828582
Avg aggreg:  542.496369401

默认的阈值是 max_word=5, min_freq=0.00005, min_entropy=2.0, min_aggreg=50

$ cd data/
$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_result
......
=== TOTAL TRUE WORDS RECALL:    0.637
=== TOTAL TEST WORDS PRECISION: 0.452
=== F MEASURE:  0.529
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.093
=== IV Recall Rate:     0.671

结果不是很好，说明默认的参数并不好，不适合我们这个测试文档


调整
=====

根据之前的统计结果，看到左右邻居熵的均值在 0.081 多一些，并不高，我们这里设置了 2.0，过高了，改为 0.1 试试看

聚合达到了 542.5，而我们默认参数只有 50，过小了，改为 600 看看

由于我们已经生成了统计结果文件 candidates_statistics.csv，故此这里直接使用新的参数值来过滤并更新 good_words.csv 文件即可

$ python2.7 segment_re_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_result_entropy0.1_aggreg600 5 0.00005 0.1 600

都不需要跑评估程序，只要看看 good_words.csv 中只有 281 个词，显然条件过于苛刻了

回头看，使用默认参数时，得到的 good_words.csv 共计 672 个词，其实也不多啊，难道是默认的条件也苛刻？

比较一下默认参数和上面的参数，直觉上，熵改为 0.08 看看，已经比默认的 2.0 宽松多了； 聚合的值设为 600 看来是太大了，改为 300 试试看

$ python2.7 segment_re_runner.py ./data/pku_test.utf8 ./data/pku_test_segment_result_entropy0.08_aggreg300 5 0.00005 0.08 300

$ wc -l good_words.csv

563 个好词，仍然不够，继续调整，继续减少聚合值，改为 100 看看 ==> 1183 个好词，似乎好了不少

$ ./score pku_training_words.utf8 pku_test_gold.utf8 pku_test_segment_result_entropy0.08_aggreg100
=== TOTAL TRUE WORDS RECALL:    0.633
=== TOTAL TEST WORDS PRECISION: 0.456
=== F MEASURE:  0.530
=== OOV Rate:   0.058
=== OOV Recall Rate:    0.105
=== IV Recall Rate:     0.665

和默认参数结果差距不大


考虑了一下，应该是因为测试样本太少的缘故，导致词频不够 (凝固需要计算词频，词频过少会导致平均熵过大，因为所有词的出现频次都少)

理论上，应该使用大量样本来学习，学到的 good_words.csv 再来对测试样本进行分词；而不是直接使用测试样本来学习



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
