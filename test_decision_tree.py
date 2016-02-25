#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import decision_tree
import numpy as np


class DecisionTreeTest(unittest.TestCase):

    def setUp(self):
        my_data = np.genfromtxt('decision_tree_example.txt', dtype=None)
        self.rows = my_data.tolist()

    def test_uniquecounts(self):
        results = decision_tree.uniquecounts(self.rows)
        self.assertEqual(len(results), 3)

if __name__ == '__main__':
    unittest.main()
