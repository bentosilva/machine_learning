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
def segment(text, checker):
    "Return a list of possible segmentation of text"
    if not text: return [[]]
    res = []
    length = len(text)
    for first, rem in splits(text, L=10):
        if checker(first):
            for sub in segment(rem, checker):
                candidate = [first] + sub
#                if within_length_limit(length, candidate):
                res.append(candidate)
    # if no word in dict that start with text[0], leave it out
    if len(res) == 0:
        for sub in segment(text[1:], checker):
            candidate = [text[0]] + sub
#            if within_length_limit(length, candidate):
            res.append(candidate) 
    return res


def most_match(text, checker):
    if len(text) == 0:
        return []
    sp = splits(text, L=10)
    sp.reverse()
    # 最大匹配，故此从长到短来循环
    for first, rem in sp:
        if checker(first):
            return [first] + most_match(rem, checker)
    return [text[0]] + most_match(text[1:], checker)
    
    

# NOT a good strategy
#def within_length_limit(length, candidate):
#    return True
#    if length >= 30 and len(candidate) > length * 3 / 4:
#        return False
#    else:
#        return True

def splits(text, L=20):
    "Return a list of all possible (first, rem) pairs, len(first)<=L."
    return [(text[:i+1], text[i+1:])
            for i in range(min(len(text), L))]


#### Support functions, used to check if a term is in dictionary

class Pdist(dict):
    def __init__(self, data=[]):
        for row in data:
            self[row[0]] = 1
    "Suppose it is always True if len(key) == 1"
    def __call__(self, key):
        return self.has_key(key) 

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    for line in codecs.open(name, "r", "utf-8"):
        yield line.strip().split(sep)


