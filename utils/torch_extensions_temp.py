from __future__ import absolute_import
import torch
from collections import OrderedDict

__all__ = ['StateDictTools']

class StateDictTools:
    
    @classmethod
    def get_block(cls, state_dict, layer, block):
        items = []
        for k in state_dict:
            if k.startswith('layer%i.%i' % (layer, block)):
                items.append((k, state_dict[k]))
        if items:
            return OrderedDict(items)
        raise KeyError("Block not found!")
    
    @classmethod
    def insert_before(cls, state_dict, l, b, new_block):
        new_dict = OrderedDict()
        
        # copy the layers up to the desired block
        for k in state_dict:
            if not k.startswith('layer%i.%i' % (l, b)):
                new_dict.__setitem__(k, state_dict[k])
            else:
                first_k = k
                break
                
        # insert the new layer
        for k in new_block:
            if not k.startswith('layer%i' % l):
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
            if splits[0] == 'layer%i' % l:
                # increment block indices for inserted layer
                b = int(splits[1]) + 1
                splits[1] = str(b)
            k_ = '.'.join(splits)
            new_dict.__setitem__(k_, state_dict[k])
            
        return new_dict
    
    @classmethod
    def duplicate_block(cls, state_dict, l, b):
        return cls.insert_before(state_dict, l, b, cls.get_block(state_dict, l, b))
        
    @classmethod
    def duplicate_blocks(cls, state_dict, l, b_list=None):
        new_dict = state_dict
        offset_list = [b + i for i, b in enumerate(b_list)]
        for b in offset_list:
            new_dict = cls.duplicate_block(new_dict, l, b)
        return new_dict

    @classmethod
    def duplicate_layer(cls, state_dict, l):
        block_indices = set()
        for k in state_dict:
            if k.startswith('layer%i' % l):
                block_indices.add(int(k.split('.')[1]))
        block_indices = sorted(block_indices)

        for i in range(len(block_indices)-1):
            assert block_indices[i] == block_indices[i+1] - 1, "block indices not consecutive"

        return cls.duplicate_blocks(state_dict, l, block_indices)
            
