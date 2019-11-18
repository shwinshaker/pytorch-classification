from __future__ import absolute_import
import torch
import os
from collections import OrderedDict
from . import Logger
from . import reduce_list
from copy import deepcopy
import pickle

__all__ = ['StateDict', 'ModelArch', 'ChunkSampler']

class StateDict:

    """
    parallel model: prefix = module.
    cpu model: prefix = ''
    """

    prefix = '' # 'module.'
    l = 0 # 1
    b = 1 # 2

    num_layers = 3

    def __init__(self, model, operation='duplicate', atom='block'):
        self.state_dict = model.state_dict()
        self.best_state_dict = None

        assert operation == 'duplicate', 'operation %s not supported yet' % operation
        self.operation = operation

        assert atom in ['block', 'layer', 'model']
        self.atom = atom

    def update(self, epoch, is_best, model):
        self.state_dict = model.state_dict()
        if is_best:
            self.best_state_dict = model.state_dict()

    def get_block(self, l, b):
        items = []
        for k in self.state_dict:
            if k.startswith('%slayer%i.%i.' % (self.prefix, l+1, b)):
                items.append((k, self.state_dict[k]))
        if items:
            return OrderedDict(items)
        raise KeyError("Block not found! %i-%i" % (l, b))

    def insert_before(self, l, b, new_block):
        new_dict = OrderedDict()

        # copy the layers up to the desired block
        for k in self.state_dict:
            if not k.startswith('%slayer%i.%i.' % (self.prefix, l+1, b)):
                new_dict.__setitem__(k, self.state_dict[k])
            else:
                first_k = k
                break

        # insert the new layer
        for k in new_block:
            if not k.startswith('%slayer%i.' % (self.prefix, l+1)):
                raise ValueError('todo: Block should be renamed if insert to different stage..')
            assert(k not in new_dict), "inserted block already exists before!"
            new_dict.__setitem__(k, new_block[k])

        # copy the rest of the layer
        skip = True
        for k in self.state_dict:
            if k != first_k and skip:
                continue
            if skip:
                skip = False
            splits = k.split('.')
            if splits[self.l] == 'layer%i' % (l+1):
                # increment block indices for inserted layer
                b = int(splits[self.b]) + 1
                splits[self.b] = str(b)
            k_ = '.'.join(splits)
            new_dict.__setitem__(k_, self.state_dict[k])

        self.state_dict = new_dict

        # return new_dict

    def duplicate_block(self, l, b):
        print('> now duplicate %i-%i' % (l,b))
        self.insert_before(l, b, self.get_block(l, b))

    def duplicate_blocks(self, pairs):

        if not pairs:
            return

        # sort out based on layer
        block_indices = [[] for _ in range(self.num_layers)]
        for l, b in pairs:
            block_indices[l].append(b)
        
        # offset sequence
        def offset(li):
            return [b + i for i, b in enumerate(sorted(li))]
        block_indices = [offset(layer) for layer in block_indices]

        # duplicate
        for l, layer in enumerate(block_indices):
            for b in layer:
                print('now duplicating: layer %i - block %i' % (l, b))
                self.duplicate_block(l, b)
                # test consecutive indices
            self.get_block_indices_in_layer(l)

    def duplicate_layer(self, l):
        pairs = [(l, b) for b in self.get_block_indices_in_layer(l)]
        self.duplicate_blocks(pairs)

    def duplicate_layers(self, layers):
        for l in layers:
            self.duplicate_layer(l)

    def duplicate_model(self):
        self.duplicate_layers(range(self.num_layers))

    def grow(self, indices=None):

        # assert atom == grow_atom, 'ensure atoms are the same!'

        if self.atom == 'blocks':
            self.duplicate_blocks(indices)
            return 

        if self.atom == 'layer':
            self.duplicate_layers(indices)
            return 

        if self.atom == 'model':
            self.duplicate_model()
            return 

        raise KeyError('atom %s  not supported' % atom)


    def get_block_indices_in_layer(self, l):
        block_indices = []
        for k in self.state_dict:
            if k.startswith('%slayer%i.' % (self.prefix, l+1)):
                block_indices.append(int(k.split('.')[self.b]))
        block_indices = self.dedup(block_indices)

        assert block_indices[0] == 0, ("Block indices not start from 0", block_indices)
        for i in range(len(block_indices)-1):
            assert block_indices[i] == block_indices[i+1] - 1, ("Block indices not consecutive", block_indices, self.delute(self.state_dict))
        return block_indices

    def dedup(self, li):
        li_ = []
        for i in li:
            if i not in li_:
                li_.append(i)
        return li_

    def delute(self, dic):
        li = []
        for k in dic:
            if k.startswith('%slayer' % self.prefix):
                li.append('.'.join(k.split('.')[:self.b+1]))
        return self.dedup(li)
                

class ModelArch:

    '''
        arch: stepsize in each block in each layer
    '''
    num_layers = 3

    def __init__(self, model_name, model, depth, max_depth, dpath,
                 operation='duplicate', atom='block'):

        assert model_name.lower().startswith('resnet'), 'model_name is fixed to resnet'
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        assert (max_depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'

        blocks_per_layer = (depth - 2) // 6
        self.arch = [[1.0 for _ in range(blocks_per_layer)] for _ in range(self.num_layers)]
        self.best_arch = None
        self.max_blocks_per_layer = (max_depth - 2) // 6

        if operation == 'duplicate':
            self.grow = self._duplicate
        elif operation == 'plus':
            self.grow = self._plus

        self.err_atom = err_atom

        # state dictionary
        self.state_dict = StateDict(model, operation=operation, atom=atom)

        self.arch_history = [self.arch]

        # model architecture and stepsize logger
        self.dpath = dpath
        self.logger = Logger(os.path.join(dpath, 'arch.txt'))
        # self.logger.set_names(['epoch', *['layer%i-block%i' % (l, b) for l, layer in enumerate(self.arch) for b, _ in enumerate(layer)]])
        self.logger.set_names(['epoch', *['layer%i-#blocks' % l for l in range(self.num_layers)], '# parameters'])


    def get_num_blocks_all_layer(self, best=False):
        if best:
            num_blocks = [len(layer) for layer in self.best_arch]
        else:
            num_blocks = [len(layer) for layer in self.arch]
        assert len(num_blocks) == self.num_layers, "arch's length is not equal to number of layers. Sth goes wrong."
        return num_blocks

    def get_num_blocks_model(self, best=False):
        if best:
            num_blocks = sum([len(layer) for layer in self.best_arch])
        else:
            num_blocks = sum([len(layer) for layer in self.arch])
        return num_blocks

    def duplicate_block(self, l, b):
        '''
        Inputs:
            l: layer index
            b: block index

        Notes:
            when duplicate a block, the stepsize of this block is halved, the number of blocks in this layer is doubled
        '''

        # half the stepsize at l-b
        self.arch[l][b] /= 2

        # duplicate the block, and insert
        self.arch[l].insert(b, self.arch[l][b])


    def duplicate_blocks(self, li, limit=True):
        '''
            be careful when duplicate blocks,
            below is not correct
            for l, b in li:
                self.duplicate_block(l, b)
        '''
        if not li:
            return []

        assert(isinstance(li, list))
        assert(isinstance(li[0], tuple))
        assert(len(li[0]) == 2)

        if not limit:
            for l, b in li:
                self.arch[l][b] /= 2
                self.arch[l][b] = [self.arch[l][b]] * 2
            self.arch = [reduce_list(layer) for layer in self.arch]
            return

        blocks_all_layers = self.get_num_blocks_all_layer()
        skip_layers = set()
        filtered_duplicate_blocks = [pair for pair in li]
        for l, b in li:
            if l in skip_layers:
                print('Attempt to duplicate layer%i-block%i. Limit exceeded for arch %i-%i-%i.' % (l, b, *blocks_all_layers))
                filtered_duplicate_blocks.remove((l, b))
                continue
            if blocks_all_layers[l] + 1 > self.max_blocks_per_layer:
                print('Attempt to duplicate layer%i-block%i. Limit exceeded for arch %i-%i-%i.' % (l, b, *blocks_all_layers))
                skip_layers.add(l)
                filtered_duplicate_blocks.remove((l, b))
                continue
            self.arch[l][b] /= 2
            self.arch[l][b] = [self.arch[l][b]] * 2
            blocks_all_layers[l] += 1

        self.arch = [reduce_list(layer) for layer in self.arch]

        return filtered_duplicate_blocks

    def duplicate_layer(self, l, limit=True):
        return self.duplicate_blocks([(l, b) for b in range(len(self.arch[l]))], limit=limit)

    def duplicate_layers(self, ls, limit=True):
        return self.duplicate_blocks([(l, b) for l in ls for b in range(len(self.arch[l]))], limit=limit)

    def duplicate_model(self, limit=True):
        return self.duplicate_blocks([(l, b) for l in range(self.num_layers) for b in range(len(self.arch[l]))], limit=limit)

    def _get_indices_from_layers(self, ls):
        indices = []
        for l in ls:
            indices.extend([(l, b) for b in range(len(self.arch[l]))])
        return indices

    def grow(self, indices=None, grow_atom='block', input_atom='block', operation='duplicate', limit=True):

        # sanity check
        if grow_atom not in ['block', 'layer', 'model']:
            raise KeyError('Grow atom %s not allowed!' % atom)
        if input_atom not in ['block', 'layer', 'model']:
            raise KeyError('Input atom %s not allowed!' % atom)
        if operation not in ['plus', 'duplicate']:
            raise KeyError('Grow operation %s not allowed!' % operation)

        assert operation == 'duplicate', 'operation %s is not supported yet.' % operation

        if input_atom == 'block':
            assert indices, 'must specify block layer indices for input indices at this level!'
            assert isinstance(indices, list), 'list of indices required'
            assert isinstance(indices[0], tuple), 'tuple index required, e.g. (l, b)'
        if input_atom == 'layer':
            assert indices, 'must specify block layer indices for input indices at this level!'
            assert isinstance(indices, list), 'list of indices required'
            assert isinstance(indices[0], int), 'integer index required, e.g. (l, b)'
        

        # divergence
        if grow_atom == 'block':
            assert input_atom == 'block', 'to grow individual blocks, must specify block level indices!'
            confirmed_indices = self.duplicate_blocks(indices, limit=limit)
            self.state_dict.duplicate_blocks(confirmed_indices)
            return confirmed_indices

        if grow_atom == 'layer':
            if input_atom == 'layer':
                confirmed_indices = self.duplicate_layers(indices, limit=limit)
                # sort out this! # return indices is block level!!
                self.state_dict.duplicate_blocks(confirmed_indices)
                # have to return index level same as input
                if confirmed_indices:
                    ls, _ = zip(*confirmed_indices)
                    return sorted(list(set(ls)))

            if input_atom == 'block':
                # operate the layer based on the majority vote of the blocks
                count_err_blocks = [0,0,0]
                for l, b in indices:
                    count_err_blocks[l] += 1
                count_blocks = self.get_num_blocks_all_layer()
                layer_indices = []
                for l, (num_err, num_all) in enumerate(zip(count_err_blocks, count_blocks)):
                    if num_err / num_all > 0.5:
                        layer_indices.append(l)
                confirmed_layer_indices = self.duplicate_layers(layer_indices, limit=limit)
                self.state_dict.duplicate_layers(confirmed_layer_indices)
                return confirmed_layer_indices

        if grow_atom == 'model':
            if input_atom == 'model':
                confirmed_indices = self.duplicate_model(limit=limit)
                if confirmed_indices:
                    self.state_dict.duplicate_model()

            if input_atom == 'layer':
                # if len(indices) / self.num_layers > 0.5:
                if len(indices) > 0:
                    confirmed_indices = self.duplicate_model(limit=limit)
                    if confirmed_indices:
                        # have to return index level same as input
                        self.state_dict.duplicate_model()
                        ls, _ = zip(*confirmed_indices)
                        return sorted(list(set(ls)))
                    
            if input_atom == 'block':
                # operate the model based on the majority vote of the blocks
                num_blocks = self.get_num_blocks_model()
                if len(indices) / num_blocks > 0.5:
                    confirmed_indices = self.duplicate_model(limit=limit)
                    if confirmed_indices:
                        self.state_dict.duplicate_model()

        return []

    def update(self, epoch, is_best, model):
        if is_best:
            self.best_arch = deepcopy(self.arch)

        num_paras = sum(p.numel() for p in model.parameters())
        self.logger.append([epoch, *self.get_num_blocks_all_layer(), num_paras])
        self.arch_history.append(self.arch)

        print('    Total params: %.2fM' % (num_paras/1000000.0))

        self.state_dict.update(epoch, is_best, model)

    def __str__(self, best=False):
        ''' 
            fancy print of arch
        '''
        return '%i-%i-%i' % tuple(self.get_num_blocks_all_layer(best=best))

    def close(self):
        self.logger.close()
        with open(os.path.join(self.dpath, 'stepsize_history.pkl'), 'wb') as f:
            pickle.dump(self.arch_history, f)

    def merge_block(self, *args):
        '''
            May needs this in the future
        '''
        pass


from torch.utils.data.sampler import Sampler
class ChunkSampler(Sampler):
    """
        Samples elements sequentially from some offset. 
        Arguments:
            num_samples: # of desired datapoints
            start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples





