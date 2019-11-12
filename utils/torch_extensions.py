from __future__ import absolute_import
import torch
from collections import OrderedDict

__all__ = ['StateDictTools']

class StateDictTools:

    """
    parallel model: prefix = module.
    cpu model: prefix = ''
    """

    prefix = '' # 'module.'
    l = 0 # 1
    b = 1 # 2

    @classmethod
    def get_block(cls, state_dict, layer, block):
        items = []
        for k in state_dict:
            if k.startswith('%slayer%i.%i.' % (cls.prefix, layer, block)):
                items.append((k, state_dict[k]))
        if items:
            return OrderedDict(items)
        print(list(state_dict.keys()))
        raise KeyError("Block not found! %i-%i" % (layer, block))

    @classmethod
    def insert_before(cls, state_dict, l, b, new_block):
        new_dict = OrderedDict()

        # copy the layers up to the desired block
        for k in state_dict:
            if not k.startswith('%slayer%i.%i.' % (cls.prefix, l, b)):
                new_dict.__setitem__(k, state_dict[k])
            else:
                first_k = k
                break

        # insert the new layer
        for k in new_block:
            if not k.startswith('%slayer%i.' % (cls.prefix, l)):
                raise ValueError('todo: Block should be renamed if insert to different stage..')
            assert(k not in new_dict), "inserted block already exists before!"
            new_dict.__setitem__(k, new_block[k])

        # copy the rest of the layer
        skip = True
        for k in state_dict:
            if k != first_k and skip:
                continue
            if skip:
                skip = False
            splits = k.split('.')
            if splits[cls.l] == 'layer%i' % l:
                # increment block indices for inserted layer
                b = int(splits[cls.b]) + 1
                splits[cls.b] = str(b)
            k_ = '.'.join(splits)
            new_dict.__setitem__(k_, state_dict[k])

        return new_dict

    @classmethod
    def duplicate_block(cls, state_dict, l, b):
        return cls.insert_before(state_dict, l, b, cls.get_block(state_dict, l, b))

    @classmethod
    def duplicate_blocks(cls, state_dict, l, b_list):
        new_dict = state_dict
        offset_list = [b + i for i, b in enumerate(b_list)]
        for b in offset_list:
            new_dict = cls.duplicate_block(new_dict, l, b)
            # test consecutive indices
            print('now duplicating: layer %i - block %i' % (l, b))
            cls.get_block_indices_in_layer(new_dict, l)

        return new_dict

    @classmethod
    def duplicate_layer(cls, state_dict, l):
        block_indices = cls.get_block_indices_in_layer(state_dict, l)
        return cls.duplicate_blocks(state_dict, l, block_indices)

    @classmethod
    def get_block_indices_in_layer(cls, state_dict, l):
        block_indices = []
        for k in state_dict:
            if k.startswith('%slayer%i.' % (cls.prefix, l)):
                block_indices.append(int(k.split('.')[cls.b]))
        block_indices = cls.dedup(block_indices)

        assert block_indices[0] == 0, ("Block indices not start from 0", block_indices)
        for i in range(len(block_indices)-1):
            assert block_indices[i] == block_indices[i+1] - 1, ("Block indices not consecutive", block_indices, cls.delute(state_dict))
        return block_indices

    @classmethod
    def dedup(cls, li):
        li_ = []
        for i in li:
            if i not in li_:
                li_.append(i)
        return li_

    @classmethod
    def delute(cls, dic):
        li = []
        for k in dic:
            if k.startswith('%slayer' % cls.prefix):
                li.append('.'.join(k.split('.')[:cls.b+1]))
        return cls.dedup(li)
                



