# encoding: utf-8

from news_loader import make_news_data
import pandas as pd
import numpy as np
import jieba
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from gensim.models import word2vec


def prepare_data():
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


def tfidf_classify(df):
    # TFIDF 把分好词的句子转化为权重向量，得到训练数据 x & y
    # 由于分好词的句子中，各个分词通过 ' ' 空格相隔，这个英语的分词逻辑一致，TfidfVectorizer 按空格进行分词划分
    # ngram_range=(1, 1) 就是每个按空格相隔的分词只取 ngram=1，也就是和 jieba 的分词结果保持一致
    # min_df=2 忽略出现两次以下的分词；
    # max_features=10000 表示最多取 10000 个词计算权重，也就是结果中每个句子转化为的向量最多 10000 维
    vect = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=10000)
    xvec = vect.fit_transform(df.seg_word)
    y = df.label

    # 使用多分类的 Naive Bayes 算法训练
    train_X, test_X, train_y, test_y = train_test_split(xvec, y , train_size=0.7, random_state=1)
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    pre = clf.predict(test_X)
    print metrics.classification_report(test_y, pre)


def w2v_classify(df):
    txtlist = [sent.split() for sent in df.seg_word.values()]
    num_features = 100    # w2v 100 维度的 embedding
    min_occurence = 10    # 少于 10 次出现的词会被丢弃掉
    sample = 1e-5         # 采样的阈值, 一个词语在训练样本中出现的频率越大，那么就越会被采样
    model = word2vec.Word2Vec(txtlist, workers=4, sample=sample, size=num_features, min_count=min_occurence, window=20, iter=20)
    # TODO. 此处可以使用 Word2Vec 中的方法，把得到的 model 保存为文件
    print model.syn0.shape  # w2v 训练好的语料矩阵，词数量 X 词维度(100)



def try_w2v_model(model):
    # 相似词
    for w in model.most_similar(u'互联网'):
        print w[0], w[1]


if __name__ == '__main__':
    df = prepare_data()
    tfidf_classify(df)
