#!/usr/bin/env python
# encoding: utf-8

import codecs
import numpy as np
from collections import namedtuple as nt

Term = nt('Term', ['freq', 'left', 'right', 'aggreg', 'inner', 'score'])


def load_data(fname='candidates_statistics.csv'):
    terms = {}
    with codecs.open(fname, 'r', 'utf-8') as fp:
        for line in fp:
            term, freq, left, right, aggreg, inner, score = line.strip().split('\t')
            terms[term] = Term._make([float(freq), float(left), float(right), float(aggreg), float(inner), float(score)])
    return terms


def filter_column(terms, column_name):
    result = {}
    for text, term in terms.iteritems():
        result[text] = getattr(term, column_name)
    return result


def statistics(values):
    print "min: {}".format(np.amin(values))
    print "max: {}".format(np.amax(values))
    print "diff: {}".format(np.ptp(values))
    print "median: {}".format(np.median(values))
    print "mean: {}".format(np.mean(values))
    print "std: {}".format(np.std(values))
    print "var: {}".format(np.var(values))
    print "corrcoef: {}".format(np.corrcoef(values))
    print "histogram: {}".format(np.histogram(values, 20))


def test():
    terms = load_data()
    print "number of whole candidate terms: {}".format(len(terms))

    for col in ['freq', 'left', 'right', 'aggreg', 'inner', 'score']:
        print "---------------------------- {} -----------------------------".format(col)
        statistics(filter_column(terms, col).values())


if __name__ == '__main__':
    test()
