from __future__ import absolute_import
import torch
import os
from . import Logger

import statistics

__all__ = ['Trigger']

class Trigger(object):
    def __init__(self, hooker, smooth='median', lastn=5, thresh=1.1):
        """
            todo: pass in errs only, infer structures
        """

        if smooth == 'median':
            self.smooth = statistics.median
        elif smooth == 'mean':
            self.smooth = statistics.mean
        else:
            raise KeyError("Estimation method not allowed!")

        self.lastn = lastn
        self.thresh = thresh

        self.history = [[[] for _ in range(len(layer)-1)] for layer in hooker]
        self.baseErrs = [[] for layer in hooker]

    def feed(self, errs):
        assert len(errs) == len(self.history), "err: %i, history: %i" % (len(errs), len(self.history))

        # merge into history, and if exceeds capacity, dequeue
        for l, layer in enumerate(errs):
            # if history cleared before, reshape based on new errs
            if not self.history[l]:
                self.history[l] = [[e] for e in layer]
                continue
            for b, err in enumerate(layer):
                self.history[l][b].append(err)
                if len(self.history[l][b]) > self.lastn: 
                    self.history[l][b].pop(0)

        # set base, happens when initialize, or after model updated
        # it's possible that only one layer's base needs to be updated
        for l, layer in enumerate(self.history):
            if not self.baseErrs[l] and len(self.history[l][0]) == self.lastn:
                # errs[l] = [], as we reset in trigger
                print("Base error set for layer %i. Now allow to grow." % (l+1))
                for e, errs in enumerate(layer):
                    self.baseErrs[l].append(self.smooth(errs))

        return self.triggered()

    def triggered(self):
        layerDouble = []
        for l, layer in enumerate(self.history):
            if not self.baseErrs[l]:
                # err base not yet set
                continue
            for b, errs in enumerate(layer):
                # print(self.baseErrs, l, b)
                if self.smooth(errs) / self.baseErrs[l][b] >= self.thresh:
                    print("Truncated error exceeds at Layer %i - Block %i " % (l+1, b+2))
                    layerDouble.append(l+1) # convert to actual name layer, starts from 1
                    # clear history of updated layer
                    self.history[l] = []
                    # clear base errors of updated layer
                    self.baseErrs[l] = []
                    # break this layer
                    break

        return layerDouble

    def close(self):
        self.logger.close()

