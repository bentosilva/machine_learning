# encoding: utf-8

import sys
import chinese
import hmm_trainer
import uniout 
import codecs

ht = hmm_trainer.hmm_trainer()
ht.train('../icwb2-data/training/pku_training.utf8')

stopper = {u'，':1, u'。':1, u'；':1, u'？':1, u'！':1, u'：':1, u'\n':1}

with codecs.open(sys.argv[1], 'r', 'utf-8') as fin:
    with codecs.open(sys.argv[2], 'w', 'utf-8') as fout:
        for line in fin:
            start = -1
            for i in range(len(line)):
                if not stopper.has_key(line[i]):
                    if start == -1:
                        start = i
                else:
                    if start != -1:
                        # 看到一个stopper，那么结束前面的连续汉字字符串；
                        # 因为没有 trim 过，故此 line 一定以 \n 结尾，就是说，每行的最后一个字符一定是stopper
                        text = line[start:i]
                        tags, score = ht.segment(text)
                        # 前后都加一个空格
                        fout.write(u' ' + ht.print_seg(text, tags) + u' ')
                        start = -1
                    fout.write(line[i])


