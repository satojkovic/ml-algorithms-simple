#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool


class SimpleMapReduce(object):

    def __init__(self, map_func, reduce_func, num_workers=None):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.pool = Pool(num_workers)

    def partition(self, mapped_values):
        pass

    def __call__(self, inputs, chunksize=1):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
