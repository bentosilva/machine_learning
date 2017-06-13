#!/usr/bin/env python
# encoding: utf-8

from math import log10, floor


ideal_distribution = [0.30103, 0.176091, 0.124939, 0.09691, 0.0791812, 0.0669468, 0.0579919, 0.0511525, 0.0457575]


def get_leading_digit(number):
    """ 1 ~ 9 """
    expo = floor(log10(number))
    return int(number * 10 ** -expo)
