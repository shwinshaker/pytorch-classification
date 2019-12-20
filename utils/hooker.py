from __future__ import absolute_import
import torch
import os
import statistics
import pickle
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
        # print(type(self.input))
        # print(len(self.input))
        # print(self.input[0].size())
        self.output = output

    def unhook(self):
        self.hooker.remove()


class LayerHooker(object):
    def __init__(self, layer, layername=None, skipfirst=True,
                 scale_stepsize=False, device=None):

        self.hookers = []
        for block in layer:
            self.hookers.append(Hooker(block))

        if not layername:
            self.layername = ''
        else:
            self.layername = layername

        if skipfirst:
            self.start_block = 1
        else:
            self.start_block = 0

        self.scale_stepsize = scale_stepsize

        self.device = device

    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)

    def get_activations(self, arch):
        '''
            It's very weird that input is a tuple including `device`, but output is just a tensor..
        '''
        activations = []

        # if orignial model, the residual of the first block can't be calculated
        for hooker in self.hookers[self.start_block:]:
            activations.append(hooker.input[0].detach())
        activations.append(hooker.output.detach())

        # force to one device to avoid device inconsistency
        if self.device:
            activations = [act.to(self.device) for act in activations]

        residuals = []
        for b, (input, output) in enumerate(zip(activations[:-1], activations[1:])):
            res = output - input
            if self.scale_stepsize:
                res /= arch[b]
            residuals.append(res)

        accelerations = []
        for last, now in zip(residuals[:-1], residuals[1:]):
            accelerations.append(now - last)

        return activations, residuals, accelerations

    def draw(self, arch):
        activations, residuals, accelerations = self.get_activations(arch)

        # activation norm
        act_norms = []
        for activation in activations:
            act_norms.append(torch.norm(activation).item())

        # residual norm
        res_norms = []
        for residual in residuals:
            res_norms.append(torch.norm(residual).item())

        # acceleration norm
        acc_norms = []
        for acceleration in accelerations:
            acc_norms.append(torch.norm(acceleration).item())

        return act_norms, res_norms, acc_norms

    def close(self):
        for hooker in self.hookers:
            hooker.unhook()


class ModelHooker(object):
    def __init__(self, model_name, dpath, resume=False, atom='block', scale_stepsize=False, scale=True, device=None):
        self.dpath = dpath

        self.atom = atom
        self.scale = scale
        self.scale_stepsize = scale_stepsize

        self.skipfirst=True
        if model_name.startswith('transresnet'):
            self.skipfirst=False

        self.history_norm = []

        self.logger = Logger(os.path.join(dpath, 'Avg_truncated_err.txt'))
        if not resume:
            self.logger.set_names(['epoch', 'layer1', 'layer2', 'layer3'])

        self.device = device

    def hook(self, model):
        self.layerHookers = []
        # for key in model._modules:
        for key in model.module._modules:
            if key.startswith('layer'):
                # self.layerHookers.append(LayerHooker(model._modules[key], layername=key, skipfirst=self.skipfirst, scale_stepsize=self.scale_stepsize))
                self.layerHookers.append(LayerHooker(model.module._modules[key], layername=key, skipfirst=self.skipfirst, scale_stepsize=self.scale_stepsize, device=self.device))

    def __len__(self):
        return len(self.layerHookers)

    def __iter__(self):
        return iter(self.layerHookers)

    def draw(self, epoch, archs):
        norms = []
        err_norms = []
        for layerHooker, arch in zip(self.layerHookers, archs):
            act_norms, res_norms, acc_norms = layerHooker.draw(arch)
            norms.append([act_norms, res_norms, acc_norms])

            # scale acceleration by residuals
            # scale residual by activations
            if self.scale:
                acc_norms = [2 * acc / (res0 + res1) for acc, res0, res1 in zip(acc_norms, res_norms[:-1], res_norms[1:])]
                # res_norms = [2 * res / (act0 + act1) for res, act0, act1 in zip(res_norms, act_norms[:-1], act_norms[1:])])

            err_norms.append(acc_norms)
        # print(err_norms)
        avg_err_norms = [statistics.mean(errs) for errs in err_norms]
        # print(avg_err_norms)
        avg_avg_err_norm = statistics.mean([e for errs in err_norms for e in errs])

        self.history_norm.append(norms)
        self.logger.append([epoch, *avg_err_norms])

        if self.atom == 'block':
            return err_norms
        elif self.atom == 'layer':
            return avg_err_norms
        elif self.atom == 'model':
            return avg_avg_err_norm
        else:
            raise KeyError('atom %s not supported!' % self.atom)

    def close(self):
        for layerHooker in self.layerHookers:
            layerHooker.close()
        self.logger.close()
        with open(os.path.join(self.dpath, 'norm_history.pkl'), 'wb') as f:
            pickle.dump(self.history_norm, f)


