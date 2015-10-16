# encoding: utf-8

import sys
import chinese
import hmm_trainer
import ngrams
import uniout 
import codecs

def find_best(text, ht, checker):
    if len(text) >= 40:
        return ngrams.most_match(text, checker)

    candidates = ngrams.segment(text, checker)
    for key in ngrams.segment.memo.keys():
        del(ngrams.segment.memo[key])

    if len(candidates) == 1:
        return candidates[0]
    else:
        return max(candidates, key=rate_cmp(text, ht))


def rate_cmp(sentence, ht):
    def rate(candidate):
        tags = ht.sentence_to_tags(candidate)
        return ht.rate(sentence, tags)
    return rate


ht = hmm_trainer.hmm_trainer()
ht.train('../icwb2-data/training/pku_training.utf8')

Pw  = ngrams.Pdist(ngrams.datafile('../icwb2-data/gold/pku_training_words.utf8'))
#Pw  = ngrams.Pdist(ngrams.datafile('../ngrams/ciku.txt'))

#stopper = {u'，':1, u'。':1, u'；':1, u'？':1, u'！':1, u'：':1, u'\n':1}

with codecs.open(sys.argv[1], 'r', 'utf-8') as fin:
    with codecs.open(sys.argv[2], 'w', 'utf-8') as fout:
        for line in fin:
            start = -1
            for i in range(len(line)):
                if chinese.is_chinese(line[i]):
                    if start == -1:
                        start = i
                else:
                    if start != -1:
                        # 看到一个stopper，那么结束前面的连续汉字字符串；
                        # 因为没有 trim 过，故此 line 一定以 \n 结尾，就是说，每行的最后一个字符一定是stopper
                        text = line[start:i]
                        candidate = find_best(text, ht, Pw)
                        fout.write(u' ' + u' '.join(candidate) + u' ')
                        sysencoding = sys.getfilesystemencoding()
                        print (' '.join(candidate)).encode(sysencoding)
                        start = -1
                    fout.write(line[i])


