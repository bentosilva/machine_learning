# encoding: utf-8

import sys
import codecs
import jieba


def load_file(filename):
    doc = u''
    with codecs.open(filename, 'r', 'utf-8') as f:
        doc = u''.join(f.readlines())
    return doc


def init_jieba():
    jieba.dt.check_initialized()
    with codecs.open('new_words', 'r', 'utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                jieba.add_word(w)


def dump_file(filename, segments):
    with codecs.open(filename, 'w', 'utf-8') as f:
        f.write(u" ".join(segments))


if __name__ == '__main__':
    doc = load_file(sys.argv[1])
    init_jieba()
    if sys.argv[3] == 'new':
        # 新词发现
        segments = jieba.cut(doc)
    else:
        # 精确模式
        segments = jieba.cut(doc, cut_all=False)
    dump_file(sys.argv[2], segments)
