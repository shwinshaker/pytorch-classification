from __future__ import absolute_import
import torch
import os
from . import Logger

__all__ = ['Hooker', 'LayerHooker', 'ModelHooker']

class Hooker(object):
    '''
        forward (activation) / backward (gradient) tracker
    '''
    def __init__(self, block):
        self.hooker = block.register_forward_hook(self.hook)

    def hook(self, block, input, output):
        self.input = input
        self.output = output

    def unhook(self):
        self.hooker.remove()


class LayerHooker(object):
    def __init__(self, layer, dpath, layername=None):
        self.hookers = []
        for block in layer:
            self.hookers.append(Hooker(block))

        if not layername:
            self.layername = ''
        else:
            self.layername = layername

        fpath = os.path.join(dpath, 'norm(%s).txt' % self.layername)
        self.logger = Logger(fpath)
        activations = ['activation(%i)' % i for i in range(len(self.hookers)+1)]
        residuals = ['residual(%i)' % i for i in range(len(self.hookers))]
        accelerations = ['acceleration(%i)' % i for i in range(len(self.hookers)-1)]
        self.logger.set_names(activations + residuals + accelerations)

    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)

    def get_activations(self):
        '''
            It's very weird that input is a tuple including `device`, but output is just a tensor..
        '''
        activations = []
        for hooker in self.hookers:
            # print(self.layername, type(hooker.output), hooker.input[0].size())
            activations.append(hooker.input[0].detach())
        # print(self.layername, type(hooker.output), hooker.output.size())
        activations.append(hooker.output.detach())

        residuals = []
        for input, output in zip(activations[:-1], activations[1:]):
            residuals.append(output - input)

        accelerations = []
        for last, now in zip(residuals[:-1], residuals[1:]):
            accelerations.append(now - last)

        return activations, residuals, accelerations

    def draw(self):
        activations, residuals, accelerations = self.get_activations()

        norms = []
        # activation norm
        for activation in activations:
            norms.append(torch.norm(activation))
        # residual norm
        for residual in residuals:
            norms.append(torch.norm(residual))
        # acceleration norm
        for acceleration in accelerations:
            norms.append(torch.norm(acceleration))

        return norms

    def draw_errs(self):
        norms = []
        _, _, accelerations = self.get_activations()
        for acceleration in accelerations:
            norms.append(torch.norm(acceleration).item())
        return norms

    def output(self):
        self.logger.append(self.draw())

    def close(self):
        # todo
        # self.logger.plot()
        self.logger.close()
        for hooker in self.hookers:
            hooker.unhook()


class ModelHooker(object):
    def __init__(self, model, dpath):
        self.dpath = dpath

        self.layerHookers = []
        for key in model._modules:
            if key.startswith('layer'):
                self.layerHookers.append(LayerHooker(model._modules[key], dpath, layername=key))

    def __len__(self):
        return len(self.layerHookers)

    def __iter__(self):
        return iter(self.layerHookers)

    def draw_errs(self):
        norms = []
        for layerHooker in self.layerHookers:
            norms.append(layerHooker.draw_errs())
        return norms

    def output(self):
        for layerHooker in self.layerHookers:
            layerHooker.output()

    def close(self):
        for layerHooker in self.layerHookers:
            layerHooker.close()


