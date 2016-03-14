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
    doc = load_file(sys.argv[1])
    ws = Words('', max_word=int(sys.argv[3]), min_freq=float(sys.argv[4]), min_entropy=float(sys.argv[5]), min_aggreg=float(sys.argv[6]))
    ws.train_from_candidates_file()
    sg = Segmentor(ws)
    segments = sg.run(doc)
    dump_file(sys.argv[2], segments)
