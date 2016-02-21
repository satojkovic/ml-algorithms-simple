#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import log
from collections import defaultdict
from PIL import Image, ImageDraw


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
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)

    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def buildtree(rows, scoref=entropy):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(column_count):
        column_values = defaultdict(int)
        for row in rows:
            column_values[row[col]] += 1
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0],
                            value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    if tree.results is not None:
        print str(tree.results)
    else:
        print str(tree.col) + ':' + str(tree.value) + '? '
        print indent + 'T->',
        printtree(tree.tb, indent + ' ')
        print indent + 'F->',
        printtree(tree.fb, indent + ' ')


def getwidth(tree):
    if tree.tb is None and tree.fb is None:
        return 1
    else:
        return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb is None and tree.fb is None:
        return 0
    else:
        return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawnode(draw, tree, x, y):
    if tree.results is None:
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        draw.text((x-20, y-10),
                  str(tree.col) + ':' + str(tree.value),
                  (0, 0, 0))

        draw.line((x, y, left + w1/2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2/2, y + 100), fill=(255, 0, 0))

        drawnode(draw, tree.fb, left + w1/2, y + 100)
        drawnode(draw, tree.tb, right - w2/2, y + 100)

    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x-20, y), txt, (0, 0, 0))


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w/2, 20)
    img.save(jpeg, 'JPEG')


def main():
    my_data = np.genfromtxt('decision_tree_example.txt', dtype=None)
    tree = buildtree(my_data.tolist())
    printtree(tree)

    drawtree(tree, jpeg='treeview.jpg')

if __name__ == '__main__':
    main()
