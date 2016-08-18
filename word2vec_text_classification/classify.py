# encoding: utf-8

from __future__ import absolute_import
from news_loader import make_news_data
import pandas as pd
import numpy as np
from sklearn import metrics
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils


def prepare_data():
    print "----- Prepareing Data -----"
    import cPickle
    import jieba
    # load data, and make into DataFrame df
    tags, texts, uniq_labels = make_news_data()
    df = pd.DataFrame({'label': tags, 'txt': texts})

    # jieba segmentaion and serialize
    jieba.enable_parallel(4)
    df['seg_word'] = df.txt.map(lambda x: ' '.join(jieba.cut(x)))
    # 经测试，含有 utf-8 的 df，通过普通 open 打开的文件也可以正常 dump & load
    cPickle.dump(df, open('df.pickle', 'wb'))
    # df = cPickle.load(open('df.pickle', 'rb'))

    return df


def train_and_evaluate(X, y, clf):
    from sklearn.cross_validation import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=1)
    clf.fit(train_X, train_y)
    pre = clf.predict(test_X)
    print metrics.classification_report(test_y, pre)
    print metrics.confusion_matrix(test_y, pre)


def tfidf_classify(df):
    from sklearn.naive_bayes import MultinomialNB
    print "----- Classification by TF-IDF & MultinomialNB -----"
    from sklearn.feature_extraction.text import TfidfVectorizer
    # TFIDF 把分好词的句子转化为权重向量，得到训练数据 x & y
    # 由于分好词的句子中，各个分词通过 ' ' 空格相隔，这个英语的分词逻辑一致，TfidfVectorizer 按空格进行分词划分
    # ngram_range=(1, 1) 就是每个按空格相隔的分词只取 ngram=1，也就是和 jieba 的分词结果保持一致
    # min_df=2 忽略出现两次以下的分词；
    # max_features=10000 表示最多取 10000 个词计算权重，也就是结果中每个句子转化为的向量最多 10000 维
    vect = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=10000)
    xvec = vect.fit_transform(df.seg_word)
    print "(doc size X doc vector size): ", xvec.shape

    # 使用多分类的 Naive Bayes 算法训练
    clf = MultinomialNB()
    train_and_evaluate(xvec, df.label, clf)


def word2vec_transform(df, num_features=100):    # w2v 100 维度的 embedding
    print "----- Word2Vec transform -----"
    txtlist = [sent.split() for sent in df.seg_word.values]
    min_occurence = 10    # 少于 10 次出现的词会被丢弃掉
    sample = 1e-5         # 采样的阈值, 一个词语在训练样本中出现的频率越大，那么就越会被采样
    model = word2vec.Word2Vec(txtlist, workers=4, sample=sample, size=num_features, min_count=min_occurence, window=20, iter=20)

    word2vec.Word2Vec.save(model, 'w2v_model')
    #  model = word2vec.Word2Vec.load('w2v_model')

    try_w2v_model(model)
    return df, model, txtlist


def try_w2v_model(model):
    """
    本函数用于研究 word2vec 得到的 word-embedding model，对分类没有影响
    """
    # w2v 训练好的 word-embedding 矩阵，词数量 X 词维度(100)
    print "(vocabulary size, word vector size): ", model.syn0.shape

    # 相似词，查询的词需要在 model.syn0 词库中
    for w in model.most_similar(u'孩子'):
        print w[0], w[1]


def cluster_classify(model, df, txtlist):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import SGDClassifier
    print "----- Classification by clustering the word-embeddings -----"
    # cluster words by embedding vectors
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] // 20    # 除 20 取整，也即每个 clusters 平均词数量 = 20
    clustering = KMeans(n_clusters=num_clusters)
    idx = clustering.fit_predict(word_vectors)    # idx 为 model.syn0 词库中每个词所对应的 cluster id 列表

    word_centroid_map = dict(zip(model.index2word, idx))  # 得到每个词 => cluster id 字典
    word_centroid_df = pd.DataFrame(zip(model.index2word, idx))  # 得到一个 df，第一列为词，第二列为词对应的 cluster id
    word_centroid_df.columns = ['word', 'cluster']

    # check some cluster info
    for cluster in xrange(5):
        print "\nCluster: {}".format(cluster)
        words = word_centroid_df.ix[word_centroid_df.cluster == cluster, 'word'].values
        print ' '.join(words)

    # check some big cluster
    # groupby + apply ==> something like
    # cluster
    # 1         71
    # 2         25
    # ......
    # then reset_index ==>
    #     cluster   0
    # 0   1         71
    # 1   2         25
    # ......
    cluster_size = word_centroid_df.groupby('cluster').apply(lambda x: len(x.word)).reset_index()
    cluster_size.columns = ['cluster', 'word_num']
    # 10 是平均值 20 的一半，也就是说 key_cluster 去掉了只有少数词的长尾 cluster
    key_cluster = cluster_size.ix[cluster_size['word_num'] >= 10, 'cluster'].values

    # 把 df 中每个句子都按句中的词来计数 cluster，得到句子对应的 cluster vector
    train_centroids = np.zeros((len(txtlist), num_clusters), dtype='float32')   # 训练集句子个数(df 的个数) X cluster 个数
    for i, review in enumerate(txtlist):
        train_centroids[i] = create_bag_of_centroids(review, word_centroid_map, key_cluster)
    train_centroids = np.where(train_centroids > 0, 1, 0)   # 把计数转化为 0、1 特征，也就是说个数不再重要，是否存在才重要
    train_centroids_df = pd.DataFrame(train_centroids)
    # 行不变，把完全是 0 的列去掉，因为对应列为长尾列，对分类无关
    train_centroids_df = train_centroids_df.ix[:, train_centroids.sum(axis=0) != 0]
    print "(Doc size X non-trivial-cluster size): ", train_centroids_df.shape

    # 训练
    clf = SGDClassifier()
    train_and_evaluate(train_centroids_df.values, df.label, clf)


def vectorize_classify(model, df, num_features):
    from sklearn.ensemble import GradientBoostingClassifier
    print "----- Classification by vectorize document text -----"
    sent_matrix = np.zeros([df.shape[0], num_features], float)
    for i, sent in enumerate(df.seg_word.values):
        sent_matrix[i, :] = sentvec(sent, num_features, model)
    print "(Doc size X text vectorized size): ", sent_matrix.shape

    clf = GradientBoostingClassifier()
    train_and_evaluate(sent_matrix, df.label, clf)


# 将词向量平均化为文档向量，方法是把文档中的词向量(如果在词库中)直接相加然后除以总数
# 也就是相当于把文档来做词向量化
def sentvec(sent, num_features, model):
    res = np.zeros(num_features)
    words = sent.split()
    num = 0
    for w in words:
        if w in model.index2word:
            res += model[w]
            num += 1
    return res / float(num) if num > 0 else res


def create_bag_of_centroids(wordlist, word_centroid_map, key_cluster):
    """ 给定一个词列表，返回一个 vector，其每个元素为各 cluster 上词的个数 """
    num_centroids = max(word_centroid_map.values()) + 1    # 一共 num_centroids 个 cluster
    bag_of_centroids = np.zeros(num_centroids, dtype='float32')
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            # 只累积 key cluster，放弃长尾 cluster
            if index in key_cluster:
                bag_of_centroids[index] += 1
    return bag_of_centroids


def keras_classify(df):
    # 预处理，把 text 中的词转成数字编号
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing import sequence
    from keras.callbacks import EarlyStopping
    from sklearn.cross_validation import train_test_split

    print "----- Classification by Keras -----"
    max_features = 50000  # 只选最重要的词
    # Tokenizer 只能处理 str，不能处理 unicode
    textraw = map(lambda x: x.encode('utf-8'), df.seg_word.values.tolist())
    token = Tokenizer(nb_words=max_features)
    # 由于 df.seg_word 以空格相隔，故此这里 Tokenizer 直接按英文方式处理 str 即可完成分词
    token.fit_on_texts(textraw)
    # token 中记录了每个词的编号和出现次数，这里使用词编号来代替 textraw 中的词文本
    # 如 textraw = ['a b c', 'c d e f']  ==> text_seq = [[1, 2, 3], [3, 4, 5, 6]]
    text_seq = token.texts_to_sequences(textraw)
    nb_classes = len(np.unique(df.label.values))
    print "num of features(vocabulary): ", len(token.word_counts)
    print "num of labels: ", nb_classes
    max_sent_len = np.max([len(s) for s in text_seq])
    print "max length or document is: ", max_sent_len
    median_sent_len = np.median([len(s) for s in text_seq])
    print "median length or document is: ", median_sent_len

    # 这里的 df.label.values 中 values 不能忽略，否则后面 np_utils.to_categorical 时会出错
    train_X, test_X, train_y, test_y = train_test_split(text_seq, df.label.values, train_size=0.7, random_state=1)
    # 目前 train_X & test_X 仍然不是等长的，其每行都是一个 document，需要化为等长的矩阵才能训练
    seqlen = int(max_sent_len / 2 + median_sent_len / 2)
    X_train = sequence.pad_sequences(train_X, maxlen=seqlen, padding='post', truncating='post')
    X_test = sequence.pad_sequences(test_X, maxlen=seqlen, padding='post', truncating='post')
    # 把 y 格式展开为 one-hot，目的是在 nn 的最后采用 softmax
    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_test = np_utils.to_categorical(test_y, nb_classes)

    model = build_cnn_model(max_features, seqlen, nb_classes)
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    # 训练 10 轮，每轮 mini_batch 为 32，训练完调用 earlystop 查看是否已经 ok
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, validation_split=0.1, callbacks=[earlystop])
    evaluate(earlystop.model, X_test, Y_test, test_y)

    model = build_lstm_model(max_features, seqlen, nb_classes)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, validation_split=0.1)
    evaluate(model, X_test, Y_test, test_y)

    model = build_mixed_model(max_features, seqlen, nb_classes)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, validation_split=0.1)
    evaluate(model, X_test, Y_test, test_y)

    graph = build_graph_model(max_features, seqlen, nb_classes)
    graph.fit({'input': X_train, 'output': Y_train}, nb_epoch=3, batch_size=32, validation_split=0.1)
    predict = graph.predict({'input': X_test}, batch_size=32)
    predict = predict['output']
    classes = predict.argmax(axis=1)
    acc = np_utils.accuracy(classes, test_y)
    print('Test accuracy: ', acc)


def evaluate(model, X_test, Y_test, test_y):
    score = model.evaluate(X_test, Y_test, batch_size=32)
    print("Test score: ", score)
    classes = model.predict_classes(X_test, batch_size=32)
    acc = np_utils.accuracy(classes, test_y)  # 这里要用未转化为 one-hot 之前的 y
    print("Test accuracy: ", acc)


def build_cnn_model(max_features, seqlen, nb_classes):
    # CNN 模型
    print ('build cnn model ....')
    model = Sequential()
    # 词向量嵌入层，输入：词典大小=max_features=50000、词向量大小=100 类似 Word2Vec、文本长度为上面取的 seqlen
    # NN 的嵌入层也会把每个词转为一个 word-embedding，类似于学习 WOrd2Vec
    model.add(Embedding(max_features, 100, input_length=seqlen))
    model.add(Dropout(0.25))
    # 卷积层，输入：卷积核个数 200，卷积窗口大小 10
    model.add(Convolution1D(nb_filter=200, filter_length=10, border_mode='valid', activation='relu'))
    # 池化层，输入：池化窗口大小 50
    model.add(MaxPooling1D(pool_length=50))
    model.add(Flatten())
    # 全连接层，输入：隐藏层神经元个数 50
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    # 输出层
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    return model


def build_lstm_model(max_features, seqlen, nb_classes):
    print('build lstm model ...')
    model = Sequential()
    model.add(Embedding(max_features, 100, input_length=seqlen))
    model.add(LSTM(100))
    # 全连接层，输入：隐藏层神经元个数 50
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    # 输出层
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    return model


# cnn + lstm
def build_mixed_model(max_features, seqlen, nb_classes):
    print('build mixed model ...')
    model = Sequential()
    model.add(Embedding(max_features, 100, input_length=seqlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=200, filter_length=10, border_mode='valid', activation='relu'))
    model.add(MaxPooling1D(pool_length=50))
    model.add(LSTM(100))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    return model


def build_graph_model(max_features, seqlen, nb_classes):
    from keras.models import Graph
    fw = [2, 10, 5]
    pool_length = [2, 50, 10]
    print('build graph model ...')
    graph = Graph()
    graph.add_input(name='input', input_shape=(seqlen,), dtype='int')
    # Embedding
    graph.add_node(Embedding(max_features, 100, input_length=seqlen), name='embedding', input='input')
    # 卷积两个字
    graph.add_node(Convolution1D(nb_filter=200, filter_length=fw[0], activation='relu'), name='conv1', input='embedding')
    graph.add_node(MaxPooling1D(pool_length=pool_length[0], border_mode='valid'), name='pool1', input='conv1')
    graph.add_node(Flatten(), name='flat1', input='conv1')  # 为啥不是 pool1 ？？
    # 卷积 10 个字
    graph.add_node(Convolution1D(nb_filter=200, filter_length=fw[1], activation='relu'), name='conv2', input='embedding')
    graph.add_node(MaxPooling1D(pool_length=pool_length[1], border_mode='valid'), name='pool2', input='conv2')
    graph.add_node(Flatten(), name='flat2', input='conv2')  # 为啥不是 pool2 ？？
    # 卷积 5 个字
    graph.add_node(Convolution1D(nb_filter=200, filter_length=fw[2], activation='relu'), name='conv3', input='embedding')
    graph.add_node(MaxPooling1D(pool_length=pool_length[2], border_mode='valid'), name='pool3', input='conv3')
    graph.add_node(Flatten(), name='flat3', input='conv3')  # 为啥不是 pool3 ？？
    # put it all together
    graph.add_node(Dense(50, activation='relu'), name='dense1', inputs=['flat1', 'flat2', 'flat3'], merge_mode='concat')
    graph.add_node(Dropout(0.5), name='drop1', input='dense1')
    graph.add_node(Dense(nb_classes, activation='softmax'), name='softmax', input='drop1')
    graph.add_output(name='output', input='softmax')
    graph.compile('Adam', loss={'output': 'categorical_crossentropy'})
    return graph


if __name__ == '__main__':
    df = prepare_data()
    tfidf_classify(df)
    num_features = 100
    df, model, txtlist = word2vec_transform(df, num_features)
    cluster_classify(model, df, txtlist)
    vectorize_classify(model, df, num_features)
    keras_classify(df)
