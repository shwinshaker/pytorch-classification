from models.cifar import resnet
# from models.cifar import midnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# net = midnet(num_classes=10, depth=20)
net = resnet(num_classes=10, depth=20)
print(net)
print(net.__class__.__name__)
# print(dict(dict(net.layer1.named_children())['0'].bn1.named_buffers())['running_var'].detach())
# print(dict(dict(net.layer1.named_children())['0'].bn1.named_parameters())['weight'].detach())
kernel = dict(dict(net.layer1.named_children())['0'].conv1.named_parameters())['weight'].cpu().detach().numpy()
fft_coeff = np.fft.fft2(kernel, (16, 16), axes=[2, 3])
D = np.linalg.svd(fft_coeff.T, compute_uv=False, full_matrices=False)
print(D.shape)
print(np.sort(D.flatten())[-1])
_, D, _ = np.linalg.svd(fft_coeff.T, compute_uv=True, full_matrices=False)
print(np.sort(D.flatten())[-1])
print(np.max(D))
# print(dict(net.layer1.named_children())['0'].conv1.stride)
# print(dict(net.layer1.named_children())['0'].conv1.padding)
# for name, module in net.layer1.named_children():
#     print(net.__class__.__name__)
#     print(name, module)
#     module.register_forward_hook(lambda x: x)
# net = torch.nn.DataParallel(net).cuda()
# print(net._modules)
# for name, module in net.named_children():
#     print(name, module)
# names, _ = zip(*list(net.named_buffers()))
# print(names)
# for name, buf in net.named_buffers():
#     print(name, buf)
# print(net._modules.keys())
# print(net._modules['module'])
# print(net.module._modules)
# print(net.state_dict().keys())
# for block in net._modules['layer1']:
#     print(block)
    # block.register_forward_hook(lambda x: x)
# for # layer in list(net._modules.items()):
#     # print(layer)

# from utils import scheduler
# print([s for s in scheduler.__dict__ if s.islower() and not s.startswith("__") and callable(scheduler.__dict__[s])])

# print(torch.prod(torch.tensor([torch.squeeze(torch.ones(1)), torch.max(torch.tensor([0, 1]))])))

# test if module in class still share the updated parameters
# class Test:
#     def __init__(self, module):
#         self.module = module
# 
#     def print(self):
#         print(dict(self.module.named_parameters()))
# 

# m = nn.Linear(2, 3)
# u = torch.randn(4, 2)
# m.register_buffer('u', u)
# print(dict(m.named_buffers()))
# # u = dict(m.named_buffers())['u'].detach()
# # u = torch.ones(4, 2, out=u)
# # u.copy_(torch.ones(4, 2))
# delattr(m, 'u')
# print(dict(m.named_buffers()))
# # u_ = u.clone()
# # u_ = torch.ones(4, 2)
# # u.copy_(u_)

# print(dict(m.named_buffers()))
# optimizer = optim.SGD(m.parameters(), lr=0.1)
# test = Test(m)
# test.print()
# inp = torch.randn(4, 2)
# out_ = torch.ones(4, 3)
# out = m(inp)
# loss = F.binary_cross_entropy_with_logits(out, out_)
# print(loss)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# test.print()
# Test(m).print()

# a = torch.tensor([0,1,2])
# b = torch.tensor([1,2,3])
# c = torch.tensor([1,2,3])
# print(torch.cat([a,b,c], dim=0))
# print(len(torch.cat([a,b,c], dim=0)))

