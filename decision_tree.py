#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divideset(rows, column, value):
    split_function = None

    # split function for numerical value
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    # split function for other value
    else:
        split_function = lambda row: row[column] == value

    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def uniquecounts(rows):
    results = defaultdict(int)
    for row in rows:
        r = row[len(row) - 1]
        results[r] += 1
    return results


def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def main():
    my_data = np.loadtxt('decision_tree_example.txt', dtype=np.str)
    print divideset(my_data.tolist(), 2, 'yes')

if __name__ == '__main__':
    main()
