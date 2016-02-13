#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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


def main():
    my_data = np.loadtxt('decision_tree_example.txt', dtype=np.str)
    print divideset(my_data.tolist(), 2, 'yes')

if __name__ == '__main__':
    main()
