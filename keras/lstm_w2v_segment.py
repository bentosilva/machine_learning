# encoding: utf-8

import codecs
import pandas as pd
import numpy as np
from gensim.models import word2vec
import cPickle
from itertools import chain
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential  # , Graph
from keras.layers.core import Dense, Dropout, Activation  # , TimeDistributedDense, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
# from keras.regularizers import l1, l2
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D
from sklearn.cross_validation import train_test_split


def load_training_file(filename):
    # lines like ['a b c', 'd e f']
    # words like [['a', 'b', 'c'], ['d', 'e', 'f']]
    # tags like 'BMESSBES'
    lines = []
    words = []
    tags = ''
    with codecs.open(filename, 'r', 'utf-8') as fp:
        for line in fp:
            line = line.strip('\n').strip('\r')
            lines.append(line)
            ws = line.split()
            words.append(ws)
            for w in ws:
                if len(w) == 1:
                    tags += 'S'
                else:
                    tags += 'B' + 'M' * (len(w) - 2) + 'E'
    return lines, words, tags


def word_freq(lines):
    """ 返回 DataFrame，按词频倒序排列 """
    # default filter is base_filter(), which is '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # 这样的话，比如 a-b-c 不会被当作一个词，而会被当作 a b c 三个词看待
    # 另外注意，不设置上限 nb_words
    token = Tokenizer(filters='')
    # token 只能接受 str 不能接受 unicode
    token.fit_on_texts(map(lambda x: x.encode('utf-8'), lines))
    wc = token.word_counts
    df = pd.DataFrame({'word': map(lambda x: x.decode('utf-8'), wc.keys()), 'freq': wc.values()})
    df.sort('freq', ascending=False, inplace=True)
    df['idx'] = np.arange(len(wc))
    return df


def word2vec_train(corpus, epochs=20, size=100, sg=1, min_count=1, num_workers=4, window=6, sample=1e-5, negative=5):
    """
    word-embedding 维度 size
    至少出现 min_count 次才被统计，由于要和 Tokenizer 统计词频中的词一一对应，故此这里 min_count 必须为 1
    context 窗口长度 window
    sg=0 by default using CBOW； sg=1 using skip-gram
    negative > 0, negative sampling will be used
    """
    w2v = word2vec.Word2Vec(workers=num_workers, sample=sample, size=size, min_count=min_count, window=window, sg=sg, negative=negative)
    np.random.shuffle(corpus)
    w2v.build_vocab(corpus)

    for epoch in range(epochs):
        print 'epoch: ', epoch
        np.random.shuffle(corpus)
        w2v.train(corpus)
        w2v.alpha *= 0.9    # learning rate
        w2v.min_alpha = w2v.alpha
    print 'word2vec done'
    word2vec.Word2Vec.save(w2v, 'w2v_model')
    return w2v


def word2vec_order(w2v, idx2word):
    """ 按 word index 顺序保存 word2vec 的 embedding 矩阵，而不是用 w2v.syn0 """
    ordered_w2v = []
    for i in xrange(len(idx2word)):
        ordered_w2v.append(w2v[idx2word[i]])
    return ordered_w2v


def sent2veclist(sentence, word2idx, context=7):
    """
    本函数把一个文档转为数值形式，并处理未登录词和 padding
    然后把文档中的每个字取 context 窗口，然该词在窗口中间
    sentence 为词的列表，注意不是字符串；context 即窗口长度，设为奇数
    注意：转化方法为逐字，而不是逐词，这样才能和字的 tag 一一对应上
    """
    numlist = []
    for word in sentence:
        for c in word:
            numlist.append(word2idx[c if c in word2idx else u'U'])
    pad = context / 2
    numlist = [word2idx[u'P']] * pad + numlist + [word2idx[u'P']] * pad

    veclist = []
    # 文档中的第一个字 (注意不是词) ，idx=0，恰好窗口为 numlist[0:7]
    for i in xrange(len(numlist) - pad * 2):    # 注意不是 len(sentence)，因为 sentence 是 word 的集合，而不是 char 的集合
        veclist.append(numlist[i: i + context])
    return veclist


def prepare_train_data(filename, load_w2v_file=False):
    lines, words, tags = load_training_file(filename)
    freqdf = word_freq(lines)
    max_features = freqdf.shape[0]   # 词个数
    print "Number of words: ", max_features
    word2idx = dict((c, i) for c, i in zip(freqdf.word, freqdf.idx))
    idx2word = dict((i, c) for c, i in zip(freqdf.word, freqdf.idx))

    if load_w2v_file:
        w2v = word2vec.Word2Vec.load('w2v_model')
    else:
        w2v = word2vec_train(words)
    print "Shape of word2vec model: ", w2v.syn0.shape
    ordered_w2v = word2vec_order(w2v, idx2word)

    # 定义'U'为未登陆新字, 'P'为两头padding用途，并增加两个相应的向量表示
    char_num = len(ordered_w2v)
    idx2word[char_num] = u'U'
    word2idx[u'U'] = char_num
    idx2word[char_num + 1] = u'P'
    word2idx[u'P'] = char_num + 1
    ordered_w2v.append(np.random.randn(100, ))   # for u'U'
    ordered_w2v.append(np.zeros(100, ))          # for u'P'

    # 生成训练 X/y 变量
    windows = list(chain.from_iterable(map(lambda x: sent2veclist(x, word2idx, context=7), words)))
    print "Length of train X: ", len(windows)
    print "Length of train y: ", len(tags)
    cPickle.dump(windows, open('training_chars.pickle', 'wb'))
    cPickle.dump(tags, open('training_tags.pickle', 'wb'))
    return windows, tags, ordered_w2v


def train(windows, tags, ordered_w2v, batch_size=128):
    label_dict = dict(zip(['B', 'M', 'E', 'S'], range(4)))
    num_dict = {n: l for l, n in label_dict.iteritems()}
    # tags 转化为数字
    train_label = [label_dict[y] for y in tags]

    train_X, test_X, train_y, test_y = train_test_split(np.array(windows), train_label, train_size=0.8, random_state=1)
    # label num -> one-hot vector
    Y_train = np_utils.to_categorical(train_y, 4)
    Y_test = np_utils.to_categorical(test_y, 4)
    # 初始词向量
    init_weight = [np.array(ordered_w2v)]
    # 词典大小
    max_features = init_weight[0].shape[0]
    word_dim = 100
    maxlen = 7  # 即 context
    hidden_units = 100

    print('stacking LSTM ...')
    model = Sequential()
    model.add(Embedding(max_features, word_dim, input_length=maxlen))
    model.add(LSTM(output_dim=hidden_units, return_sequences=True))   # 中间层 lstm
    model.add(LSTM(output_dim=hidden_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4))    # 输出 4 个结果对应 BMES
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    print('train ...')
    model.fit(train_X, Y_train, batch_size=batch_size, nb_epoch=20, validation_data=(test_X, Y_test))

	graph = Graph()
	graph.add_input(name='input', input_shape=(maxlen,), dtype=int)
	graph.add_node(Embedding(max_features, word_dim, input_length=maxlen), name='embedding', input='input')
	graph.add_node(LSTM(output_dim=hidden_units), name='fwd', input='embedding')
	graph.add_node(LSTM(output_dim=hidden_units, go_backwards=True), name='backwd', input='embedding')
	graph.add_node(Dropout(0.5), name='dropout', input=['fwd', 'backwd'])
	graph.add_node(Dense(4, activation='softmax'), name='softmax', input='dropout')
	graph.add_output(name='output', input='softmax')
	graph.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')	
	graph.fit({'input': train_X, 'output': Y_train, batch_size=batch_size, nb_epoch=20, validation_data=({'input': test_X, 'output': Y_test}))

	import theano
	weight_len = len(model.get_weights())
	for w in model.get_weights():
		print(w.shape)
	layer = theano.function([model.layers[0].input, model.layers[3].get_output(train=False), allow_input_downcast=True)
	layer_out = layer(test_X[:10])
	layer_out.shape   # 前 10 个窗口的第 0 层输出，经过了 relu 计算, should be (10, 4)
	temp_txt = u'国家食药监总局发布通知称，酮康唑口服制剂因存在严重肝毒性不良反应，即日起停止生产销售使用。'
	temp_txt = list(temp_txt)
	temp_vec = sent2veclist(temp_txt, word2idx, context=7)
	temp_result = make_predict(temp_vec, temp_txt, model, label_dict, num_dict)


def make_predict(input_vec, input_txt, model, label_dict, num_dict):
	input_vec = np.array(input_vec)
	predict_prob = model.predict_proba(input_vec, verbose=False)
	predict_label = model.predict_classes(input_vec, verbose=False)
	# fix 一些可能的逻辑错误
	for i, label in enumerate(predict_label[: -1]):
		# 首字不可为 E/M
		if i == 0:
			predict_prob[i, label_dict[u'E']] = 0
			predict_prob[i, label_dict[u'M']] = 0
		# 前字为B，后字不可为B,S
		if lable == label_dict[u'B']:
			predict_prob[i + 1, label_dict[u'B']] = 0
			predict_prob[i + 1, label_dict[u'S']] = 0
		# 前字为E，后字不可为M,E
		# 前字为M，后字不可为B,S
		# 前字为S，后字不可为M,E
		
		predict_label[i + 1] = predict_prob[i+1].argmax()
	predict_label_new = [num_dict[x]  for x in predict_label]
	result =  [w+'/' + l  for w, l in zip(input_txt, predict_label_new)]
	return ' '.join(result) + '\n'
		


if __name__ == '__main__':
    import sys
    infile = sys.argv[1]
    windows, tags, ordered_w2v = prepare_train_data(infile, load_w2v_file=True)
    train(windows, tags, ordered_w2v, batch_size=128)
