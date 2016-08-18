文档分类预测
==================

Refer: [使用 Word2Vec 进行文本分类](http://xccds1977.blogspot.sg/2015/05/word2vec.html) [使用深度学习库keras做文本分类](http://xccds1977.blogspot.sg/2015/11/keras.html)

本文档介绍了 4 中文档分类预测的算法

1. TF-IDF + Bayes
2. Word2Vec + Cluster
3. Word2Vec + 词向量平均化
4. Keras 深度学习库文本分类

### 数据准备

准备好一批标注了分类信息的文档，分类信息列表 tags，对应的文本列表 texts

在 prepare_data() 函数中，利用 jieba 对文本分词，分词后把文本再以空格分隔串起来

最后得到一个 DataFrame, as below:
```
       label    txt                 seg_word
   0   教育     北京教育局发布      北京 教育局 发布
   1   孩子     春天孩子容易感冒    春天 孩子 容易 感冒
   .........
```
可以使用 cPickle 模块来保存和导入

### 算法一  TF-IDF + Bayes

1. 把 df.seg_word 直接塞给 TfidfVectorizer，后者通过 ngram_range=(1, 1) 直接使用空格进行分词 (类似英文)
2. 根据每个文档中的词频，以及全部语料中词的文档频率，进行 TF-IDF 计算
3. 继而把每个文档表示为非长尾词的 TF-IDF 值所组成的 vector
4. 塞给 MultinomialNB 模型进行训练和预测

### Word2Vec

把 df.seg_word 中的每个句子，按空格进行分割，得到句子所包含的分词列表

把全部句子的分词列表塞给 Word2Vec 模型进行学习，丢掉长尾词，以 20 长度为窗口，每个词的维度定为 100，得到 Word2Vec 模型 model

其中 model.syn0 为学习好的 word-embedding 矩阵，维度为 Num_of_Vocabulary X 100; model[word] 为该 word 的 vector

并可以通过 Word2Vec.save/load 函数进行保存和导入

### 算法二  Word2Vec + Cluster

1. 有了 word2vec model，我们就相当于说知道了一个词的 vector 向量，进而可以使用 KMeans 算法把 model.syn0 中词按距离聚类
2. 聚类时，K 选择为 len(vocabulary) // 20，就是说每个类平均词数量为 20，进而把词数量小于 10 个的长尾类去掉
3. 对训练语料 df 中的每个句子，逐词累积所属的类，这样句子就表达成了一个 vector，每个元素为其所对应的类在句中占有的词数
4. 这里简化一下，每个类出现的词数不重要，是否有词出现重要，也就是把权重 vector 转化为 0/1 vector，每个元素表示对应类是否在句中有词出现
5. 把最后得到的数据塞给 LinearModel 中的 SGDClassifier 进行分类训练和预测

### 算法三  Word2Vec + 词向量平均化

这个更为简单粗暴，df 中的每个句子，逐词的 vector 直接相加，然后平均化，得到一个句子 vector，维度显然同词向量维度，也是 100

然后塞给 GradientBoostingClassifier 模型进行分类训练和预测

### 算法四 Keras 深度学习

数据准备：

- 先把 df.seg_word 中每个文档转化为 utf-8 编码，然后调用 keras 的 Tokenizer 进行分词 (空格分割，类似英文)
- Tokenizer 中记录了词表 vocabulary，词频，并通过 texts_to_sequences 方法，把句子中的词转化为词的 index 编号
- 各个文档的长度不同，故此选定 (max(文档长度) + median(文档长度)) / 2 为 seqlen，把训练集和测试集中的句子化为等长 (padding和截断)
- 为了便于和 softmax 层的输出比较，通过 keras.utils.np_utils.to_category 把训练集和测试集的标签转化为 one-hot 向量

到此，训练集 X & Y 维度分别为 (训练集句子数 X seqlen) 和 (训练集句子数 X 标签数)

然后 keras 采用了 4 中模型来训练，分别为

1. CNN: Word-Embedding -> Dropout -> Convolution1D -> MaxPooling1D -> Flatten -> Dense -> Dropout -> Relu -> Dense -> Softmax
2. LSTM: Word-Embedding -> LSTM -> Dense -> Dropout -> Relu -> Dense -> Softmax
3. CNN+LSTM: Word-Embedding -> Dropout -> Convolution1D -> MaxPooling1D -> LSTM -> Dropout -> Relu -> Dense -> Softmax
4. Graph:
    input -> Word-Embedding ---> Convolution1D -> MaxPooling1D -> Flatten ---> Dense(relu) -> Dropout -> Dense(softmax) -> output
                             |-> Convolution1D -> MaxPooling1D -> Flatten -|
                             |-> Convolution1D -> MaxPooling1D -> Flatten -|
