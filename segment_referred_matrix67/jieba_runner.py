# encoding: utf-8

import sys
import codecs
import jieba


def load_file(filename):
    doc = u''
    with codecs.open(filename, 'r', 'utf-8') as f:
        doc = u''.join(f.readlines())
    return doc


def dump_file(filename, segments):
    with codecs.open(filename, 'w', 'utf-8') as f:
        f.write(u" ".join(segments))


if __name__ == '__main__':
    doc = load_file(sys.argv[1])
    if sys.argv[3] == 'new':
        # 新词发现
        segments = jieba.cut(doc)
    else:
        # 精确模式
        segments = jieba.cut(doc, cut_all=False)
    dump_file(sys.argv[2], segments)
