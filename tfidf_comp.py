#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
import math
from nltk.corpus import stopwords
import numpy as np

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
    return math.log((len(bloblist) + 1.) / (1. + sk_n_containing(word, bloblist)))


def sk_tfidf(word, blob, bloblist):
    return sk_tf(word, blob) * (1 + sk_idf(word, bloblist))


def nltk_stopwords(fileids='english'):
    return stopwords.words(fileids)


def preprocessing(data):
    processed_data = []

    # english stop words
    stop_words = nltk_stopwords('english')
    # to lowercase
    lower_data = [d.lower() for d in data]
    # remove stopwords
    for ldata in lower_data:
        processed_data.append(' '.join([lw for lw in ldata.split()
                                        if lw not in stop_words]))
    return processed_data


def normalize(scores):
    scores_norm = defaultdict(float)
    norm = np.linalg.norm(scores.values())
    for key in scores.keys():
        scores_norm[key] = scores[key] / norm
    return scores_norm


def main():
    bloblist = preprocessing(data)
    for i, blob in enumerate(bloblist):
        print 'TF-IDF in document {}'.format(i + 1)
        scores = {word: sk_tfidf(word, blob, bloblist)
                  for word in blob.split()}
        scores_norm = normalize(scores)
        sorted_words = sorted(scores_norm.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            print '\tWord: {}, TF-IDF: {}'.format(word, round(score, 5))

if __name__ == '__main__':
    main()
