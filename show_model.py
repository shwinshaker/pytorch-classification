from models.cifar import resnet
import torch

net = resnet(num_classes=10, depth=38)
# net = torch.nn.DataParallel(net).cuda()
# print(net._modules.keys())
# print(net._modules['module'])
# print(net.module._modules)
print(net.state_dict().keys())
# for block in net._modules['layer1']:
#     print(block)
    # block.register_forward_hook(lambda x: x)
# for # layer in list(net._modules.items()):
#     # print(layer)

# from utils import scheduler
# print([s for s in scheduler.__dict__ if s.islower() and not s.startswith("__") and callable(scheduler.__dict__[s])])
