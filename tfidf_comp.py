#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import math

data = ["Human machine interface for ABC computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement"]


def sk_tf(word, blob):
    return Counter(blob.split())[word]


def sk_n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.split())


def sk_idf(word, bloblist):
    return math.log((len(bloblist) + 1) / (1 + sk_n_containing(word, bloblist)))


def sk_tfidf(word, blob, bloblist):
    return sk_tf(word, blob) * (1 + sk_idf(word, bloblist))


def main():
    bloblist = data
    for i, blob in enumerate(bloblist):
        print 'Top words in document {}'.format(i + 1)
        scores = {word: sk_tfidf(word, blob, bloblist)
                  for word in blob.split()}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            print '\tWord: {}, TF-IDF: {}'.format(word, round(score, 5))

if __name__ == '__main__':
    main()
