# encoding: utf-8

import sys
import chinese
import ngrams
import uniout 

#Pw  = ngrams.Pdist(ngrams.datafile('CorpusWordlist.txt'), missingfn=ngrams.avoid_long_words)
Pw  = ngrams.Pdist(ngrams.datafile('ciku.txt'), missingfn=ngrams.avoid_long_words)

sysencoding = sys.getfilesystemencoding()
utext = sys.argv[1].decode(sysencoding)
text = ''.join([w if chinese.is_chinese(w) else '' for w in utext])

seg = ngrams.segment(text, Pw)
print seg

