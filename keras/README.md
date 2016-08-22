脚本介绍
===========

- lstm_gen_text.py -- 旧版本 Keras example 中的代码，加入我的注释；注意新版本中该脚本代码有了一些变化，名字也改为 lstm_text_generation.py
- lstm_segment_v${i}.py + lstm_segment_runner_v${i}.py -- 参考 lstm_gen_text.py 实现了 LSTM 的中文分词，把窗口预测词的逻辑改为窗口预测 BEMS 标记
- lstm_w2v_segment.py -- 另一个版本的深度学习分词


lstm_segment_v${i}.py 利用 LSTM 中文分词
===========================================

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


lstm_w2v_segment.py 利用深度学习 + Word2Vec 进行分词
=======================================================

参考文档:  [基于深度LSTM的中文分词](http://xccds1977.blogspot.sg/2015/11/blog-post_25.html)

和前面的分词有一些不同之处：

- 语料的处理
	+ 前面的版本把整个语料合成一个大的字符串，只在最前面和最后面添加 padding 字符；这样的话，中间的 \r\n 换行字符都保留下来
	+ 这里把语料中的每行单独处理，每行的前面和后面添加 padding 字符，这样就不再需要处理换行符，直接 strip 掉即可

- 每个"字"的处理
	+ 前面的版本采用的是 one-hot 向量化，字符向量的维度为训练语料中独立字的个数，达到万的量级，故此训练数据非常大，导致了内存不足而需要分批处理
	+ 这里使用了 word-embedding 的方法，使用了一个 Embedding 层，把每个独立字符转化为一个默认 100 维的向量，这样训练数据要小的多了
		原参考 link 中首先使用 gensim 对训练文本进行分词，然而训练得到的 w2v 模型却没有真正利用起来；完全可以不做分词，因为 Embedding 层本质上就是学习了一个关于独立字符的 embedding 向量表达；
		另外，gensim 学到的 w2v 模型其实是关于训练样本中独立"词"的向量，而不是字符的，故此 w2v 模型就算学习出来了，也用不到训练语料的字符上


注意事项： 我写了一个 word_freq 函数，使用 keras 的 Tokenizer 来计算训练语料中的“词”频率
  	然而，其实程序整个算法都是基于字符的，无论是字符频率还是判断未登录字符，都不是基于词的
  	和前面 word-embedding 部分的介绍类似，训练语料中的“词”其实对算法无意义，只决定了词中字符的 BMES 标识而已
	使用 nltk 库中的 Text 配合 FreqDist 可以得到字符级的频率


流程和逻辑：
```
prepare_train_data(filename)
	load_training_file(filename)
		lines - ['天气 不错', '出去 玩', ..]
			char_freq(lines)
				freqdf - DataFrame {header: 'word', 'freq', 'idx', values: [ ..., ['天', 53, 15], ..., ['气', 39, 31], ....}
			max_features - freqdf.shape[0] 全部独立字符数
			word2idx - 实际应为 char2idx，字符 => idx，并加入 padding 和 未登录字向量
			idx2word - 实际应为 idx2char，idx => 字符，并加入 padding 和 未登录字向量
		words - [ ['天气', '不错'], ['出去', '玩'], ..]
			sent2veclist(word, word2idx, context=7)
				windows - [ [pad,pad,pad,15,31,...], [pad,pad,15,31, ...], ...]
		tags - 'BEBEBES...'
	return windows, tags, word2idx

run(windows, tags, word2idx, test_file, batch_size=128)
	label_dict - label -> idx
	num_dict - idx -> label
	windows ==> train_X/test_X
	tags transformed by label_dict ==> train_y/test_y one-hot ==> Y_train, Y_test
	Stacking-LSTM:  Embedding -> LSTM -> LSTM -> Dropout -> Dense -> Activation
	Graph:  input -> Embedding ---> LSTM -----> Dropout -> Dense(softmax) -> output
                                |-> LSTM  ->|
	predict sample & test_file
```                      

