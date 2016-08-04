利用 LSTM 中文分词
===================

### ver.1

参考文档 [基于Deep Learning的中文分词尝试](https://mp.weixin.qq.com/s?__biz=MzA4OTk5OTQzMg==&mid=2449231335&idx=1&sn=d3ba98841e85b7cea0049cc43b3c16ca)

通过修改 lstm_gen_text.py (新版本 keras 中的 lstm_text_generation.py) 的代码逻辑来实现

运行时报错
```
Using Theano backend.
corpus length: 1864560
chars length: 4701
total tags: 4
nb sequences: 621520
Vectorization...
Traceback (most recent call last):
  File "runner.py", line 66, in <module>
      workflow('./data/pku_training.utf8', './data/pku_test.utf8', './output')
  File "runner.py", line 57, in workflow
      X, y, chars, tags, char_indices = pretraining(train_text, train_tags, maxlen, step)
  File "/home/cuiyong/workspace/myproj/learning/keras/lstm_segment.py", line 56, in pretraining
      X = np.zeros((len(windows), maxlen, len(chars)), dtype=np.bool)
MemoryError
```

np.zeros((len(windows), maxlen, len(chars))) 维度为 621520 * 7 * 4701 = 0.6 M * 7 * 4.7K = 20G，就算每个 float 占 4 个字节，那么也要 80 G

maxlen = 7 和 one-hot 4701 基本不会变，那么按 X 只占 1G 内存来算，1G / 7 / 4.7K / 4 = 7.6 K

故此一次性 X 最多能放进 7.6 K 个切分的测试样本，我们可以把 7.6 K 个切分样本作为一个 Batch，把测试文件能切分出的全部样本按 Batch 分批 train

调整文件，得到 ver.2

### ver.2
