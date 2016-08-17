# encoding: utf-8

from news_loader import make_news_data
import pandas as pd
import numpy as np
from sklearn import metrics
from gensim.models import word2vec


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


if __name__ == '__main__':
    df = prepare_data()
    tfidf_classify(df)
    num_features = 100
    df, model, txtlist = word2vec_transform(df, num_features)
    cluster_classify(model, df, txtlist)
    vectorize_classify(model, df, num_features)
