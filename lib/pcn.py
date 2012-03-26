# -*- coding: utf-8 -*-

# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008



from numpy import *

class pcn:
    """ 単純パーセプトロン """
    
    def __init__(self, inputs, targets):
        """ Constructor """
        # setup network size
        if ndim(inputs):
            self.nIn = shape(inputs)[1]
        else:
            self.nIn = 1

        if ndim(targets):
            self.nOut = shape(targets)[1]
        else:
            self.nOut = 1
        
        self.nData = shape(inputs)[0]
        
        # 重みをランダムな数値で初期化
        self.weights = random.rand(self.nIn+1, self.nOut)*0.1-0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        """学習フェーズ"""
        # バイアスノードを追加
        inputs = concatenate((inputs, -ones((self.nData, 1))), axis=1)
        # 学習
        change = range(self.nData)

        for n in range(nIterations):
            
            self.outputs = self.pcnfwd(inputs);
            self.weights += eta * dot(transpose(inputs), targets-self.outputs)

            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]

            print "Iteration:" , n
            print self.weights

        print "Final outpus are:"
        print self.outputs

    def pcnfwd(self, inputs):
        """再現フェーズ"""
        
        outputs = dot(inputs, self.weights)

        # Threshold the outputs
        return where(outputs > 0, 1, 0)




        

