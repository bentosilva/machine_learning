# encoding: utf-8

import math
import re
import codecs
from collections import namedtuple, defaultdict as dd
import jieba


class Algorithm(object):
    @staticmethod
    def binary_divide(text):
        """
            given abcd, return [[a, bcd], [ab, cd], [abc, d]]
            given ab return [[a, b]]
            given a return []
        """
        return [[text[0: i], text[i:]] for i in range(1, len(text))]

    @staticmethod
    def sorted_substrs_indices(text, n):
        """
            get all substrings' indices with length eq or less than n
            this func is used to generate all possible valid word from corpus
        """
        length = len(text)
        res = []
        # substring 的开头索引
        for i in xrange(0, length):
            # substring 的结尾索引
            for j in xrange(i + 1, min(i + n + 1, length + 1)):
                res.append([i, j])
        # 按 word 排序，但是返回序列号，因为后面需要通过序列号找到左右邻居
        return sorted(res, key=lambda (i, j): text[i: j])

    @staticmethod
    def entropy(wdict):
        """
            entropy of a word dict - sigma(-p[i] * log(p[i])) for i in dict
            the larger entropy is, the larger information of list contains
            when the list is the neighbor of a word, the larger the possibility to be good segment
        """
        total = float(sum(wdict.values()))
        return sum([-1 * c * math.log(c / total) / total for w, c in wdict.items()])

    @staticmethod
    def aggregation(word, allwords):
        """
            word is the given Word instance to calculate internal aggregation
            allwords is a dict for the whole corpus, keyed by word's text, valued by Word instance
            显然，要在所有的词都被统计过之后才能运行这个算法
        """
        parts = Algorithm.binary_divide(word.text)
        """
            为什么取最小？举个例子，binary_divide(牡丹花) = [[牡丹, 花], [牡, 丹花]]
            显然应该使用第一组的结果，因为第一组的 divide 更有代表性；然而显然第一组得到的结果更小 (分母的频率大)
            结果越大，那么说明组合在一起的概率比分开的概率之积的比值越大，成词可能越大
        """
        return math.log(min([word.freq / allwords[pl].freq / allwords[pr].freq for pl, pr in parts])) if len(parts) > 0 else 1.0


class Word(object):
    """
        word info including frequency, neighbors, inner aggregation
    """
    def __init__(self, text):
        super(Word, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = dd(int)
        self.right = dd(int)
        self.aggreg = 0.0

    def meet(self, left, right):
        self.freq += 1
        if left:
            self.left[left] += 1
        if right:
            self.right[right] += 1

    def statistics(self, corpus_length):
        self.freq /= corpus_length
        self.left = Algorithm.entropy(self.left)
        self.right = Algorithm.entropy(self.right)


class Words(object):
    def __init__(self, training_doc, batch_size=10000, max_word=5, min_freq=0.00005, min_entropy=2.0, min_aggreg=50):
        """
            max_word - 一个词最多5个字符长
        """
        super(Words, self).__init__()
        self.batch_size = batch_size
        self.max_word = max_word
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_aggreg = min_aggreg
        self.doc = training_doc

    def train(self):
        print "counting training doc ..."
        pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        candidates = {}
        doc_length = 0
        # 注意，doc 初始为一个空格，doc[0] 位置的字符不作为每次遍历的目标
        # 而只是作为 doc[1] 的左邻居，这样避免训练下一个批次时，丢掉了起点的左邻
        doc = u' '
        line_cnt = 0
        with codecs.open(self.doc, 'r', 'utf-8') as f:
            for line in f:
                line = re.sub(pattern, '', line)
                doc_length += len(line)
                doc += line
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print "{} lines processed".format(line_cnt)
                    # if line_cnt == 110000:
                    #     break
                # 每 batch_size 个汉字处理一次
                if len(doc) < self.batch_size:
                    continue
                length = len(doc)
                # 从 1 开始遍历，目的是保留上次遍历留下来的左邻居
                # 不取到 length，目的是保证从这一轮循环的每个起点起，
                # 都能取到 self.max_word 长度的字串
                # 比如 length = 10 ==> 0 1 2 3 4 5 6 7 8 9
                # self.max_word = 5 ==> i 最多取到 4，这样可以取到
                # "45678" 字串，而且能取到右邻居 ‘9’
                for i in xrange(1, length - self.max_word):
                    for j in xrange(i + 1, i + self.max_word + 1):
                        text = doc[i: j]
                        if text not in candidates:
                            candidates[text] = Word(text)
                        candidates[text].meet(doc[i - 1: i], doc[j: j + 1])
                # 本批次处理完毕，准备处理下一批次，那么前面处理过的字符可以删掉了
                # 但是，最后的一个字符不能删，因为要作为下一个批次的 doc[0]，即左邻居
                doc = doc[length - self.max_word - 1:]
        # 循环完毕，那么 doc 中剩下一些不到 self.batch_size 长的字符，需要做一下处理
        length = len(doc)
        # 同样，跳过 doc[0]
        for i in xrange(1, length):
            for j in xrange(i + 1, min(i + self.max_word + 1, length + 1)):
                text = doc[i: j]
                if text not in candidates:
                    candidates[text] = Word(text)
                candidates[text].meet(doc[i - 1: i], doc[j: j + 1])
        # 计算 freq 和左右邻熵
        print "making statistics ..."
        for word in candidates.values():
            word.statistics(doc_length)
        # 至此，全部 freq 都被计算完毕，可以计算凝固度了
        print "calculating aggregations ...."
        for text, word in candidates.items():
            if len(text) < 2:
                continue
            word.aggreg = Algorithm.aggregation(word, candidates)
        # 到这里，单个的词已经无用了，后面词库只记录双字以上的词
        self.words = sorted([word for text, word in candidates.items() if len(text) > 1], key=lambda v: v.freq, reverse=True)
        # 一些统计数据
        total = float(len(self.words))
        print "Avg len: ", sum([len(w.text) for w in self.words]) / total
        print "Avg freq: ", sum([w.freq for w in self.words]) / total
        print "Avg left ent: ", sum([w.left for w in self.words]) / total
        print "Avg right ent: ", sum([w.right for w in self.words]) / total
        print "Avg aggreg: ", sum([w.aggreg for w in self.words]) / total
        # 保存当前结果
        with codecs.open("candidates_statistics.csv", "w", "utf-8") as f:
            for w in self.words:
                f.write(u"{}\t{}\t{}\t{}\t{}\n".format(w.text, w.freq, w.left, w.right, w.aggreg))
        # 过滤其中满足条件的词
        self.filter(self.words)

    def train_from_candidates_file(self):
        """
            如果之前训练过了一个语料之后，想调整参数再看看结果
            那么就可以使用这个函数，直接使用之前得到的 candidates file
            就避免了重复的计数、统计和计算
        """
        TupleWord = namedtuple("TupleWord", ['text', 'freq', 'left', 'right', 'aggreg'])

        def yield_words_from_file():
            with codecs.open("candidates_statistics.csv", "r", "utf-8") as f:
                for line in f:
                    try:
                        text, freq, left, right, aggreg = line.strip().split('\t')
                    except Exception, e:
                        print e
                        print line
                    yield TupleWord(text, float(freq), float(left), float(right), float(aggreg))
        self.filter(yield_words_from_file())

    def filter(self, words):
        """
            只取至少两个字组成的词
        """
        jieba.dt.check_initialized()
        with codecs.open("good_words.csv", "w", "utf-8") as f:
            self.good_words = {}
            for w in words:
                if w.text not in jieba.dt.FREQ and len(w.text) > 1 and w.aggreg > self.min_aggreg and\
                        w.freq > self.min_freq and w.left > self.min_entropy and\
                        w.right > self.min_entropy:
                    f.write(u"{}\t{}\t{}\t{}\t{}\n".format(w.text, w.freq, w.left, w.right, w.aggreg))
                    self.good_words[w.text] = 1


class Segmentor(object):
    def __init__(self, trained_words):
        """
            trained_words - 已经做过训练的 Words 实例
        """
        super(Segmentor, self).__init__()
        self.trained_words = trained_words

    def run(self, sentence):
        i = 0
        res = []
        while i < len(sentence):
            # 首先考虑两个字以上组成的词，最后在看单字，单子并不在 trained_words.good_words 中
            for j in range(2, self.trained_words.max_word + 1) + [1]:
                if j == 1 or sentence[i: i + j] in self.trained_words.good_words:
                    res.append(sentence[i: i + j])
                    i += j
                    break
        return res


if __name__ == '__main__':
    import sys
    doc = sys.argv[1]
    # ws = Words(doc)
    # ws.train()
    ws = Words('', min_freq=0, min_entropy=0.93, min_aggreg=6.79271256)
    ws.train_from_candidates_file()
    # sg = Segmentor(ws)
    # print sg.run(''.join(codecs.open(doc, 'r', 'utf-8').readlines()))
