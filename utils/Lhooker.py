#!./env python

class Hooker:
    """
        hook on single node, e.g. conv, bn, relu
    """

    def __init__(self, node):
        self.hooker = node.register_forward_hook(self.hook)
        self.input = None
        self.output = None

        if self.__class__.__name__.startswith('Conv'):
            self.output = self.__conv_output
        elif self.__class__.__name__.startswith('BatchNorm'):
            self.output = self.__bn_output
        else:
            self.output = lambda: 1 # Lipschitz constant 1 for any other nodes

    def hook(self, node, input, output):
        self.input = input
        self.output = output

    def unhook(self):
        self.hooker.remove()
        self.__remove_buffers()

    def __conv_output(self):
        # only when needed, i.e. after the entire validation batch, do power iteration and compute spectral norm, to gain efficiency

        buffers = dict(self.named_buffers())
        if 'u' not in buffers:
            assert 'v' not in buffers
            self.__init_buffers(self.input.size())

    def __init_buffers(self, input_shape):
        # input shape is of length 4, includes an additional batch
        assert len(input_shape) == 4
        u_shape = (1, *input_shape[1:])
        print(u_shape) # should be (3, 224, 224) for the first one

    def __remove_buffers(self):


class BlockHooker:
    # named_children -> immediate children

    def lip():
        lip = 1
        for hooker in self.hookers():
            lip_ = hooker.output()
            lip *= lip
        # TODO: some log
        return lip

class LayerHooker:
    # named_children -> immediate children
    # no need to skip first?

class ModelHooker:
    # named_children -> immediate children





