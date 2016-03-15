# encoding: utf-8

import sys
import re
import codecs


def filter(infile, outfile):
    pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
    with codecs.open(infile, 'r', 'utf-8') as fin, codecs.open(outfile, 'w', 'utf-8') as fout:
        for line in fin:
            fout.write(re.sub(pattern, '', line))

if __name__ == '__main__':
    filter(sys.argv[1], sys.argv[2])
