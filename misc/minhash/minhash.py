# encoding=utf-8

import random

def MinHash(corpus, k):
    words = {}
    # copus 词典，每个词只出现一次
    for w in corpus:
        # words 以每个词典中的词为 key
        # 每个key维护一个list，维护k次随机排列中该key的序号
        words[w] = []
    for i in range(k):
        shuffled = list(corpus)
        random.shuffle(shuffled)
        for j in range(len(shuffled)):
            words[shuffled[j]].append(j)

    def hash(document):
        total = 0.
        vals = [-1]*k
        for w in document:
            # 即使有词典中不存在的词也没关系，直接忽略
            if w in words:
                m = words[word]
                for i in range(k):
                    # 每个文档在每次随机排列中，只取一个值
                    # 就是该文档在该排列中最靠前的词的位置
                    if vals[i] == -1 or m[i] < vals[i]:
                        vals[i] = m[i]
        # 最后，加起来取平均
        return sum(vals) / k 

    return hash
                

