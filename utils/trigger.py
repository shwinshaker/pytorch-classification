from __future__ import absolute_import
import torch
import os
from . import Logger

import statistics

__all__ = ['Trigger']

#todo after doubled, hooker needs to be reset


class Trigger(object):
    def __init__(self, hooker, smooth='median', lastn=5, thresh=1.1):

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

        fpath = os.path.join(hooker.dpath, 'Truncated_err.txt')
        self.logger = Logger(fpath)
        names = ['err(%i-%i)' % (l, b) for l, h in enumerate(hooker.layerHookers) for b in range(len(h)-1)]
        self.logger.set_names(names)

    def set_names(self, hooker):
        names = ['err(%i-%i)' % (l, b) for l, h in enumerate(hooker.layerHookers) for b in range(len(h)-1)]
        self.logger.set_names(names)

    def feed(self, hooker):
        errs = hooker.draw_errs()
        assert len(errs) == len(self.history), "err: %i, history: %i" % (len(errs), len(self.history))

        print(errs)
        self.logger.append([e for l in errs for e in l])

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

    def triggered(self):
        layerDouble = []
        for l, layer in enumerate(self.history):
            if not self.baseErrs[l]:
                # history in short to set base
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

