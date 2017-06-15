#!/usr/bin/env python
# encoding: utf-8

from math import log10, floor
import numpy as np
from scipy.stats import ks_2samp
from scipy import spatial


ideal_distribution = np.array([0.30103, 0.176091, 0.124939, 0.09691, 0.0791812, 0.0669468, 0.0579919, 0.0511525, 0.0457575])


def get_leading_digit(number):
    """ 1 ~ 9 """
    if number == 0:
        return 0
    expo = floor(log10(number))
    return int(number * 10 ** -expo)


def pearson_correlation(dist):
    """ -1 ~ 1 正负相关性 """
    return np.corrcoef(ideal_distribution, dist)[0, 1]


def kstest(dist):
    """ If the K-S statistic is small or the p-value is high, then we cannot
        reject the hypothesis that the distributions of the two samples are the same.
        p-value: 发生第一类错误(弃真)的几率，其阈值 α 可取单尾也可取双尾，ks_2samp 只实现了双尾
        α 一般规定=0.05或=0.01，其意义为：假设检验中如果拒绝时，发生Ⅰ型错误的概率为5%或1%，即100次拒绝的结论中，平均有5次或1次是错误的。
        这里直接返回 p-value 就好了，K-S 统计量虽然也行，但是不好评判
    """
    return ks_2samp(ideal_distribution, dist).pvalue


def cosine_similarity(dist):
    """
    余弦相似度，角度越接近，余弦越大，余弦距离越小。故此，用 1 减去余弦距离，让结果越大越好
    但是，在测试中，我们看到，其实不是很相似的两个分布，也可能有较大的余弦值，故此这个 measurement 并不好
    """
    return 1 - spatial.distance.cosine(dist, ideal_distribution)


def euclidean_normed(dist):
    """
    正则化欧几里德距离，类似余弦相似度，需要用 1 减一下；同样，类似的，基于距离的算法并不好找阈值，不如基于 p-value 的方法
    """
    dst = spatial.distance.euclidean(dist, ideal_distribution)
    norm = pow(sum(ideal_distribution[:-1] * ideal_distribution[:-1]) + pow(1 - dist[-1], 2), 0.5)
    return 1 - dst / norm


if __name__ == '__main__':
    target_dists = np.array([[0.3, 0.18, 0.125, 0.09, 0.08, 0.065, 0.06, 0.055, 0.045],
                             [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12]])
    for dist in target_dists:
        print "------------- "
        print dist
        print "pearson correlation: {}".format(pearson_correlation(dist))
        print "ks test p-value: {}".format(kstest(dist))
        print "cosine similarity: {}".format(cosine_similarity(dist))
        print "euclidean distance normed: {}".format(euclidean_normed(dist))

    import os
    import re
    import codecs
    from collections import defaultdict as dd

    def find_files(root):
        cnt = 0
        for folder, _, files in os.walk(root):
            for f in files:
                f = os.path.join(folder, f)
                _, ext = os.path.splitext(f)
                if ext == '.md':
                    cnt += 1
                    if cnt % 1000 == 0:
                        print "number done: {}".format(cnt)
                    yield f

    def find_numbers(text):
        pat = r'(\d{1,3}(,\d{3})+(\.\d+)?|\d*\.?\d+)'
        for match in re.findall(pat, text):
            yield float(match[0].replace(',', ''))

    scores = dd(list)
    for f in find_files('../../segment_referred_matrix67/'):
        with codecs.open(f, 'r', 'utf-8') as fp:
            numbers = np.array([0.0] * 10)
            text = fp.read()
            for number in find_numbers(text):
                numbers[get_leading_digit(number)] += 1
            total = sum(numbers[1:])
            if total <= 0.0:
                continue
            dist = np.array(numbers[1:]) / total
            scores[f].append(pearson_correlation(dist))
            scores[f].append(kstest(dist))
            scores[f].append(cosine_similarity(dist))
            scores[f].append(euclidean_normed(dist))
            scores[f].append(str(list(numbers)))
            scores[f].append(str(list(dist)))
    with codecs.open('benford_scores', 'w', 'utf-8') as fp:
        for f, scorelist in scores.iteritems():
            fp.write(u"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(scorelist[0], scorelist[1], scorelist[2], scorelist[3], scorelist[4], scorelist[5], f.decode('utf-8')))
