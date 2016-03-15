# encoding: utf-8

import sys
import codecs
from matrix67_segment import Words, Segmentor


def load_file(filename):
    doc = u''
    with codecs.open(filename, 'r', 'utf-8') as f:
        doc = u''.join(f.readlines())
    return doc


def dump_file(filename, segments):
    with codecs.open(filename, 'w', 'utf-8') as f:
        f.write(u" ".join(segments))


if __name__ == '__main__':
    doc_test = load_file(sys.argv[2])
    ws = Words(sys.argv[1])
    ws.train()
    sg = Segmentor(ws)
    segments = sg.run(doc_test)
    dump_file(sys.argv[3], segments)
