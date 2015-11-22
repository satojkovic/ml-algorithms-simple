#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simple_mapreduce import SimpleMapReduce


def file_to_words(filename):
    pass


def count_words(item):
    word, occurances = item
    return (word, sum(occurances))


def main():
    import operator
    import glob

    input_files = glob.glob("*.rst")

    mapper = SimpleMapReduce(file_to_words, count_words)
    word_counts = mapper(input_files)
    word_counts.sort(key=operator.itemgetter(1))
    word_counts.reverse()

    print '\nTOP 20 WORDS BY FREQUENCY\n'
    top20 = word_counts[:20]
    longest = max(len(word) for word, count in top20)
    for word, count in top20:
        print '%-*s: %5s' % (longest+1, word, count)


if __name__ == '__main__':
    main()
