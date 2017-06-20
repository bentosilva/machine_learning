#!/usr/bin/env python
# encoding: utf-8

""" reference:
    https://arxiv.org/pdf/1201.3432.pdf
"""

from math import log10, floor
import numpy as np
from scipy import spatial
from scipy.stats import chisquare


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


def pearson_chisquare(dist, N):
    """ 对于 8 (9-1) 个自由度，5% significant level，critical value of chi-square is  15.507
        这里返回 critical value - chi_square statistics
        如果返回值大于 0，则通过测试，95% 满足 benford's law，且越大越好；否则，未通过测试
    """
    return 15.507 - chisquare(dist, ideal_distribution)[0] * N


def confidence_intervals(counts, N):
    """ 对于 5% 置信区间，z-value = 1.96
        返回观察值在 1~9 都落在置信区间的个数；越大越好，9 表示全部数字都落在置信区间
    """
    cis = 1.96 * np.sqrt((ideal_distribution * (1 - ideal_distribution)) / N) * 100
    ideal_counts = ideal_distribution * N
    return len([i for i, c in enumerate(counts) if c >= (ideal_counts[i] - cis[i]) and c <= (ideal_counts[i] + cis[i])])


def kstest(counts, N):
    """ 不要使用 scipy.stats.ks_2samp，结果不对
        按 http://epublications.bond.edu.au/cgi/viewcontent.cgi?article=1004&context=mike_steele 这个计算
        对于 5% significant level，critical value equals to 1.36   [by formula: cv(alpha) = sqrt(-0.5 * ln(alpha / 2))
        对于样本总数 N, 如果 ks-statistics > critical value / N，则拒绝 H0 假设，分布不满足 benford law
        故此，我们这里返回 (critical value / N) - ks-statistics
        如果返回值大于 0，则通过测试，95% 满足 benford's law，且越大越好；否则，未通过测试
    """
    cv = 1.36 / np.sqrt(N)
    return cv - np.max(np.abs(np.array([np.sum((counts - N * ideal_distribution)[0:i + 1]) for i in range(9)]))) / N


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
            scores[f].append(str(list(numbers[1:])))
            scores[f].append(str(list(dist)))
    with codecs.open('benford_scores', 'w', 'utf-8') as fp:
        for f, scorelist in scores.iteritems():
            fp.write(u"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(scorelist[0], scorelist[1], scorelist[2], scorelist[3], scorelist[4], scorelist[5], f.decode('utf-8')))
