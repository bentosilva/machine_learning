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

np.zeros((len(windows), maxlen, len(chars))) 维度为 621520 * 7 * 4701 = 0.6 M * 7 * 4.7K = 20G，就算每个 np.bool 占 1 个字节，那么也要 20 G

故此报内存错误


### ver.2

maxlen = 7 和 one-hot 4701 基本不会变，那么按 X 只占 1G 内存来算，1G / 7 / 4.7K  = 30.3 K

故此一次性 X 最多能放进 30.3 K 个切分的测试样本，我们可以把 30.3 K 个切分样本作为一个 Batch，把测试文件能切分出的全部样本按 Batch 分批 train

调整文件，得到 ver.2，继续尝试

运行时报错
```
python2.7 lstm_segment_runner_v2.py
Using Theano backend.
corpus length: 1864560                      <去掉空格的训练文本长度>
chars length: 4701                          <训练文本中独立字数>
total tags: 4                               <BEMS>
nb sequences: 621520                        <间隔为 3 进行切分得到的窗口数>
batchsize based on mem:  30388              <30.3k 个窗口为一个 batch>
no. of batches:  21                         <共计 21 个batch>
Build model...

--------------------------------------------------
Iteration:  1
   |__ batch:  0
Vectorization...
Epoch 1/1
30388/30388 [==============================] - 436s - loss: 0.7246
   |__ batch:  1
Vectorization...
Epoch 1/1
30388/30388 [==============================] - 491s - loss: 0.4345
   |__ batch:  2
Vectorization...
Traceback (most recent call last):
   File "lstm_segment_runner_v2.py", line 68, in <module>
      workflow('./data/pku_training.utf8', './data/pku_test.utf8', './output')
   File "lstm_segment_runner_v2.py", line 60, in workflow
      for i, model in enumerate(segmentor.training(batchsize, batchnum)):
   File "/home/cuiyong/workspace/myproj/learning/keras/lstm_segment_v2.py", line 100, in training
      X, y = self.vectorize_per_batch(batchsize, batch_idx)
   File "/home/cuiyong/workspace/myproj/learning/keras/lstm_segment_v2.py", line 83, in vectorize_per_batch
      X = np.zeros((len(windows), self.maxlen, len(self.chars)), dtype=np.bool)
MemoryError
```
第一次迭代的第 3 个 batch 都还没有处理完，就又内存问题了。

一看还是 X = np.zeros 函数的问题，似乎说明前两次的结果没有释放？导致内存累积，然而不应该啊，X, y 讲道理是要被回收的啊

后来经过调小一个批次的总内存量，从 1G 调到 400M，发现 3 个 batch 之后内存就会稳定下来不会再增长，因而很可能是 lstm 训练导致的，而不是内存泄漏

那么，就把内存调小吧，400M 一个批次的话，训练中内存会稳定在 996M；原来一个G的话，内存报错了，如果不报错，估计一下应该在 2.7G 左右


顺手，调整一下代码逻辑，不再限制测试文本中的字一定在训练文本中了，这个限制过于强了

由于目前 one-hot 编码的维度是训练文本中独立的字数 len(self.chars) 决定的，那么如果确定不在训练文本中的字的 one-hot 编码呢？

- 要么 one-hot 都是 0
- 要么大改一下，使 one-hot 编码的维度按训练文本+测试文本中的独立字数决定

这里用了前一种，因为后一种的要求也过于强了，如果新预测一个文件，而文件中有之前没有的字，就只能重新训练了 ...
