from __future__ import absolute_import
import torch
import os
from . import Logger
from . import reduce_list

import statistics

__all__ = ['Trigger', 'MinTrigger']

class Trigger(object):
    def __init__(self, smooth='median', window=5, backtrack=10, thresh=1.1):
        """
            todo: pass in errs only, infer structures
        """

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        else:
            raise KeyError("Estimation method not allowed!")

        self.window = window
        self.backtrack = backtrack
        self.thresh = thresh

        self.history = []
        self.baseErrs = []

    def feed(self, errs):
        # assert len(errs) == len(self.history), "err: %i, history: %i" % (len(errs), len(self.history))

        # merge into history, and if exceeds capacity, dequeue
        if not self.history:
            self.history = [[[] for e in layer] for layer in errs]
        for l, layer in enumerate(errs):
            for b, err in enumerate(layer):
                self.history[l][b].append(err)

        # set base, happens when initialize, after model updated, or more than backtrack epochs elapsed
        if not self.baseErrs:
            self.baseErrs = [[None for e in layer] for layer in errs]
        for l, layer in enumerate(self.history):
            for b, err in enumerate(layer):
                if not self.baseErrs[l][b] and len(self.history[l][b]) >= self.window:
                    # should be exactly ==, relax to >=
                    self.baseErrs[l][b] = self.smooth(self.history[l][b][-self.window:])
                    print("Base error initialized for layer%i - block%i ." % (l, b))
                if len(self.history[l][b]) >= self.backtrack:
                    self.baseErrs[l][b] = self.smooth(self.history[l][b][-self.backtrack:-self.backtrack+self.window]) 

        print(self.baseErrs)
        for layer in self.history:
            for errs in layer:
                if len(errs) >= self.window:
                    print(self.smooth(errs[-self.window:]), end=' ')
        print()

    def trigger(self):
        err_indices = []
        for l, layer in enumerate(self.history):
            for b, err in enumerate(layer):
                if not self.baseErrs[l][b]:
                    # err base not yet set
                    continue
                if self.smooth(self.history[l][b][-self.window:]) / self.baseErrs[l][b] > self.thresh:
                    err_indices.append((l, b))
                    # clear history of updated layer
                    # self.history[l][b] = []
                    # clear base errors of updated layer
                    # self.baseErrs[l][b] = None

        if err_indices:
            print("Truncated error exceeds at blocks", sorted(err_indices))
        return sorted(err_indices)

        # todo - propose candidate layer based on average layer errr
    
    def _layer_average(self):
        '''
            average the err norm across layer for each epoch
            and see if the avg err norm in layer is increasing
                seems complicated and not very intuitive, let's use majority vote
        '''
        pass

    def update(self, block_indices):
        # now only update history for confirmed duplicate
        # excessive duplicate will not cut the history
        for l, b in block_indices:
            if b == len(self.history[l]):
                # there are n blocks, but only n-1 errs
                self.history[l].append([])
                self.baseErrs[l].append(None)
                continue
            self.history[l][b] = [[], []] # replace by two placeholders
            self.baseErrs[l][b] = [None, None]

        self.history = [reduce_list(layer, order=2) for layer in self.history]
        self.baseErrs = [reduce_list(layer) for layer in self.baseErrs]


    def close(self):
        pass
        # self.logger.close()


class MinTrigger(object):
    
    def __init__(self, smooth='median', thresh=1.1, atom='layer'):
        """
            todo: pass in errs only, infer structures
        """

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        else:
            raise KeyError("Estimation method not allowed!")

        self.thresh = thresh
        self.atom = atom
        assert atom == 'layer', 'block wise not supported for minTrigger!'
        
        self.minErrs = []
        self.curErrs = []

    def feed(self, errs):
        if self.minErrs:
            assert len(errs) == len(self.minErrs), "err: %i, min errs: %i" % (len(errs), len(self.minErrs))

        if not self.minErrs:
            self.minErrs = [e for e in errs]
            return

        for l in range(len(self.minErrs)):
            if not self.minErrs[l]:
                self.minErrs[l] = errs[l]
            else:
                self.minErrs[l] = min(self.minErrs[l], errs[l])

        self.curErrs = [e for e in errs]

        print('cur err: ', self.curErrs)
        print('min errs: ', self.minErrs)

    def trigger(self):

        err_indices = []
        for l, (e, me) in enumerate(zip(self.curErrs, self.minErrs)):
            if e / me > self.thresh:
                err_indices.append(l)

        print("Truncated error exceeds at layers", sorted(err_indices))
        return sorted(err_indices)

    def update(self, err_indices):
        # now only update history for confirmed duplicate
        # excessive duplicate will not cut the history
        for l in err_indices:
            self.minErrs[l] = None

    def close(self):
        pass
        # self.logger.close()
