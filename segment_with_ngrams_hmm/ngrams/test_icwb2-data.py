# encoding: utf-8

import sys
import chinese
import ngrams
import uniout 
import codecs

Pw  = ngrams.Pdist(ngrams.datafile('ciku.txt'), missingfn=ngrams.avoid_long_words)

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
                        # 看到一个非汉字字符，那么结束前面的连续汉字字符串；
                        # 因为没有 trim 过，故此 line 一定以 \n 结尾，就是说，每行的最后一个字符一定不会是汉字
                        text = line[start:i]
                        seg = ngrams.segment(text, Pw)
                        # 前后都加一个空格
                        fout.write(u' ' + u' '.join(seg) + u' ')
                        start = -1
                    fout.write(line[i])


