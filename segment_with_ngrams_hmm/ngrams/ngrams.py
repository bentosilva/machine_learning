# encoding: utf-8

import re, string, operator
from collections import defaultdict
from math import log10
import codecs

def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        key = ','.join(filter(lambda x: isinstance(x, str) or isinstance(x, unicode), args))
        if key not in table:
            table[key] = f(*args)
        return table[key]
    fmemo.memo = table
    return fmemo

################ Word Segmentation

# use @memo decorator to apply dynamic programming
@memo
def segment(text, score_func):
    "Return a list of words that is the best segmentation of text."
    if not text: return []
    candidates = ([first]+segment(rem, score_func) for first,rem in splits(text))
    "Find best list from candidates (list of list) using Pword which scores each list"
    return max(candidates, key=Pwords(score_func))

def splits(text, L=20):
    "Return a list of all possible (first, rem) pairs, len(first)<=L."
    return [(text[:i+1], text[i+1:])
            for i in range(min(len(text), L))]

def Pwords(score_func):
    def inner(words):
        "The Naive Bayes probability of a sequence of words."
        return sum(score_func(w) for w in words)
    return inner


#### Support functions, score func (object in fact) with corpus

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key):
        if key in self: return log10(self[key]/self.N)
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    for line in codecs.open(name, "r", "utf-8"):
        yield line.split(sep)

#def avoid_long_words(key, N):
#    "Estimate the probability of an unknown word."
#    return 10./(N * 10**len(key))

def avoid_long_words(key, N):
    "Estimate the probability of an unknown word."
#    return 10./(N * 10**((len(key) + 1)**2))
#    return 1 - log10(N) - (len(key) + 1)** 2
    return len(key) * (1 - log10(N) - len(key))

#N = 1024908267229 ## Number of tokens

#Pw  = Pdist(datafile('count_1w.txt'), N, avoid_long_words)

#### segment2: second version, with bigram counts, (p. 226-227)

def cPw(score_bi, score_uni):
    def inner(word, prev):
        "Conditional probability of word, given previous word."
        try:
            return log10(score_bi[prev + ' ' + word]/float(score_uni[prev]))
        except KeyError:
            return log10(score_uni(word))
    return inner

#P2w = Pdist(datafile('count_2w.txt'), N)

# Three differences with segment():
# uses a conditional bigram language model, cPw, rather than the unigram model Pw.
# passed a single argument (the text), and the previous word as parameters
# The return value is the probability of the segmentation, followed by the list of words.
@memo
def segment2(text, prev='<S>', score_bi=None, score_uni=None):
    "Return (log P(words), words), where words is the best segmentation."
    if not text: return 0.0, []
    candidates = [combine(cPw(score_bi, score_uni)(first, prev), first, segment2(rem, first, score_bi, score_uni))
                  for first,rem in splits(text)]
    return max(candidates)

def combine(Pfirst, first, (Prem, rem)):
    "Combine first and rem results into one (probability, words) pair."
    return Pfirst+Prem, [first]+rem

