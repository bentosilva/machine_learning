# encoding=utf-8
from collections import defaultdict
import operator
import codecs
import re
from math import log10

class hmm_trainer(object):

    def __init__(self):
        self.stats = ['S', 'B', 'M', 'E',]

    def cal_pi(self, counts_pi):
        sum = reduce(operator.add, counts_pi.values(), 0)
        for s in counts_pi:
            self.params_pi[s] = counts_pi[s] / float(sum)

    def cal_a(self, counts_a):
        for sb in counts_a:
            sum = reduce(operator.add, counts_a[sb].values(), 0)
            for se in counts_a[sb]:
                self.params_a[sb][se] = counts_a[sb][se] / float(sum)

    def cal_b(self, counts_b):
        tlen = len(self.terms)
        for s in counts_b:
            # sum = sum + terms.length
            sum = reduce(operator.add, counts_b[s].values(), 0) + tlen
            # 训练好后， 分词时遇到词库中没有的词，默认返回 1 / sum
            self.params_b[s] = defaultdict(lambda :1/float(sum))
            for t in self.terms:
                # 需要对词进行平滑处理, 训练时，状态 s 下不会出现的词，平滑到 1 / sum
                self.params_b[s][t] = ((counts_b[s][t] if counts_b[s].has_key(t) else 0) + 1) / float(sum) 


    def train(self, file, delimiter=u'  '):
        self.params_pi = defaultdict(float)
        self.params_a  = defaultdict(lambda: defaultdict(float))
        self.params_b  = {}  # defaultdict(lambda: defaultdict(float))  ... defaultdict seems not work with setdefault 
        self.terms = []

        counts_pi = defaultdict(int)
        counts_a  = defaultdict(lambda: defaultdict(int))
        counts_b  = defaultdict(lambda: defaultdict(int))
        tdict = {}

        with codecs.open(file, 'r', 'utf-8') as fp:
            for line in fp:
                for sentence in re.split(u'，|。|；|？|！|：', line.strip()):
                    tag = ''  # at the begining of the sentence
                    for word in sentence.strip().split(delimiter):
                        word = word.strip()
                        if len(word) == 0:
                            continue
                        # 先不区分数字、字母、其他字符，看看情况
                        if len(word) == 1:
                            tdict[word] = 1
                            if tag == '':
                                counts_pi['S'] += 1
                            else:
                                counts_a[tag]['S'] += 1
                            counts_b['S'][word] += 1
                            tag = 'S'
                        else:
                            # the first term
                            tdict[word[0]] = 1
                            if tag == '':
                                counts_pi['B'] += 1
                            else:
                                counts_a[tag]['B'] += 1
                            counts_b['B'][word[0]] += 1
                            tag = 'B'

                            # the middle terms
                            for i in range(len(word) - 2):
                                term = word[i + 1]
                                tdict[term] = 1
                                counts_a[tag]['M'] += 1
                                counts_b['M'][term] += 1
                                tag = 'M'

                            # the last term
                            tdict[word[-1]] = 1
                            counts_a[tag]['E'] += 1
                            counts_b['E'][word[-1]] += 1
                            tag = 'E'
        self.terms = tdict.keys()
        self.cal_pi(counts_pi)
        self.cal_a(counts_a)
        self.cal_b(counts_b)

    """
        print len(self.terms)
        print counts_pi
        print self.params_pi
        print counts_a
        print self.params_a
        for k in counts_b:
            print k, len(counts_b[k])
        for k in self.params_b:
            print k, len(self.params_b[k])
    """


    # given a legal chinese sentence
    def segment(self, sentence):
        pre_step_dist = defaultdict(float)
        memo = []
        # init pre_xxx with the 1st term
        for stat in self.params_pi:
            pre_step_dist[stat] = log10(self.params_pi[stat]) + log10(self.params_b[stat][sentence[0]])

        # vetebi algorithm term by term
        slen = len(sentence)
        # 对位置循环，直到最后一个汉字
        for i in range(slen - 1):
            t = sentence[i + 1]
            cur_step_dist = defaultdict(float)
            step_dict = {}
            # 对状态循环，确定这个位置本步的状态
            for se in self.stats:
                # 对前一个汉字的状态循环，找到最大值作为本步的最大概率
                for sb in pre_step_dist:
                    if self.params_a[sb].has_key(se):
                        try:
                            p = pre_step_dist[sb] + log10(self.params_a[sb][se]) + log10(self.params_b[se][t])
                        except Exception, e:
                            print "Error:", e
                            print "a: %f    b: %f" % (self.params_a[sb][se], self.params_b[se][t])
                            print "Environment: %s %s %s" % (sb, se, t)
                            raise e
                        # 注意 p 是小于 0 的数，因为取了 log10；而 cur_step_dist 默认值是 0
                        if p > cur_step_dist[se] or cur_step_dist[se] == 0:
                            step_dict[se] = sb
                            cur_step_dist[se] = p

            pre_step_dist = cur_step_dist
            memo.append(step_dict)

        # 回溯输出分词结果，以及打分
        solution, score = sorted(pre_step_dist.items(), key=lambda x:x[1], reverse=True)[0]
        tags = [''] * (slen - 1) + [solution]
        # i from slen - 1 to 1, and memo is from slen - 2 to 0
        for i in range(slen - 1, 0, -1):
            tags[i - 1] = memo[i - 1][solution]
            solution = memo[i - 1][solution]

        return tags, score


    # given a legal chinese sentence
    def print_seg(self, sentence, tags):
        restr = ''
        str_tag = {'S': u' ', 'B': u'', 'M': u'', 'E': u' '}
        for i, t in enumerate(sentence):
            restr += t + str_tag[tags[i]]
        return restr.strip()


    # given a solution(entence and the tags), calculate score
    def rate(self, sentence, tags):
        slen = len(sentence)
        sb = tags[0]
        score = log10(self.params_pi[sb]) + log10(self.params_b[sb][sentence[0]])
        for i in range(slen - 1):
            t = sentence[i + 1]
            se = tags[i + 1]
            score += log10(self.params_a[sb][se]) + log10(self.params_b[se][t])
            sb = se 
        return score
        

    # from a segment candidate array, return the tags
    def sentence_to_tags(self, candidate):
        tags = []
        for item in candidate:
            if len(item) == 1:
                tags.append('S')
            else:
                tags += ['B'] + ['M'] * (len(item) - 2) + ['E']
        return tags

