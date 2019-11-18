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
        self.output = output

    def unhook(self):
        self.hooker.remove()


class LayerHooker(object):
    # def __init__(self, layer, dpath, layername=None, resume=False):
    def __init__(self, layer, layername=None):
        self.hookers = []
        for block in layer:
            self.hookers.append(Hooker(block))

        if not layername:
            self.layername = ''
        else:
            self.layername = layername


    def __len__(self):
        return len(self.hookers)

    def __iter__(self):
        return iter(self.hookers)

    def get_activations(self, arch, scale=True):
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
        for b, (input, output) in enumerate(zip(activations[:-1], activations[1:])):
            # residuals.append(output - input)
            '''
                It's not clear should we scale the residual by the stepsize here
            '''
            if scale:
                residuals.append((output - input)/arch[b])
            else:
                residuals.append(output - input)

        accelerations = []
        for last, now in zip(residuals[:-1], residuals[1:]):
            accelerations.append(now - last)

        return activations, residuals, accelerations

    def draw(self, arch, scale=True):
        activations, residuals, accelerations = self.get_activations(arch, scale=scale)

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

        # track the history
        # if output:
        #     self.logger.append(act_norms + res_norms + acc_norms)

        return act_norms, res_norms, acc_norms

    def close(self):
        # todo
        # self.logger.plot()
        # self.logger.close()
        for hooker in self.hookers:
            hooker.unhook()


class ModelHooker(object):
    def __init__(self, model, dpath, resume=False):
        self.dpath = dpath

        self.layerHookers = []
        for key in model._modules:
            if key.startswith('layer'):
                # self.layerHookers.append(LayerHooker(model._modules[key], dpath, layername=key, resume=resume))
                self.layerHookers.append(LayerHooker(model._modules[key], layername=key))

        self.history_norm = []

        # self.logger = Logger(os.path.join(dpath, 'Avg_truncated_err.txt'), resume=resume)
        self.logger = Logger(os.path.join(dpath, 'Avg_truncated_err.txt'))
        # activations = ['activation(%i)' % i for i in range(len(self.hookers)+1)]
        # residuals = ['residual(%i)' % i for i in range(len(self.hookers))]
        # accelerations = ['acceleration(%i)' % i for i in range(len(self.hookers)-1)]
        if not resume:
            self.logger.set_names(['epoch', 'layer1', 'layer2', 'layer3'])

    def reset(self, model):
        self.layerHookers = []
        for key in model._modules:
            if key.startswith('layer'):
                self.layerHookers.append(LayerHooker(model._modules[key], layername=key))

    def __len__(self):
        return len(self.layerHookers)

    def __iter__(self):
        return iter(self.layerHookers)

    def draw(self, epoch, archs, atom='block', scale=True):
        norms = []
        err_norms = []
        for layerHooker, arch in zip(self.layerHookers, archs):
            act_norms, res_norms, acc_norms = layerHooker.draw(arch, scale=scale)
            norms.append([act_norms, res_norms, acc_norms])
            err_norms.append(acc_norms)
        avg_err_norms = [statistics.mean(errs) for errs in err_norms]

        self.history_norm.append(norms)
        self.logger.append([epoch, *avg_err_norms])

        if atom == 'block':
            return err_norms
        elif atom == 'layer':
            return avg_err_norms
        else:
            raise KeyError('atom %s not supported!' % atom)

    # Let's now automatically record when draw errs
    # def output(self):
    #      for layerHooker in self.layerHookers:
    #         layerHooker.output()

    def close(self):
        for layerHooker in self.layerHookers:
            layerHooker.close()
        self.logger.close()
        with open(os.path.join(self.dpath, 'norm_history.pkl'), 'wb') as f:
            pickle.dump(self.history_norm, f)


