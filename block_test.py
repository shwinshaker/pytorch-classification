#!./env python
# from models.cifar.resnet import BasicBlock
import torch.nn as nn
import torch
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
            x is not residual, but out...
        """
        # out = x
        residual = x

        # # preresnet
        # residual = self.bn1(residual)
        # residual = self.relu(residual)
        # residual = self.conv1(residual)

        # residual = self.bn2(residual)
        # residual = self.relu(residual)
        # residual = self.conv2(residual)

        # resnet - scaled by 1/2, but technically not right, since the last relu is applied as $relu(out + residual)$, rather than $out + relu(residual)$
        # residual = self.conv1(residual)
        residual = self.bn1(residual)
        # residual = self.relu(residual)

        # residual = self.conv2(residual)
        # residual = self.bn2(residual)
        # residual = self.relu(residual)

        # the first block in a stage is downsampled to be consistent
        # should separate this out
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out = x + residual
        # out = self.relu(out)

        return out - x 


torch.manual_seed(0)
block = BasicBlock(8, 8)
# print(block.state_dict().keys())

# torch.manual_seed(0)
# x0 = torch.randn(1, 8, 32, 32)

# implicit initialization test
# print(block.state_dict()['bn2.weight'])
# 
# x1 = block(x0)
# print(torch.norm(x1))
# print(torch.norm(x1/2))
# 
# # scale
# for key in block.state_dict():
#     if 'weight' in key or 'bias' in key:
#         # print(key)
#         # block.state_dict()[key].data.copy_(block.state_dict()[key] * 0) # / 10) #  2 #  1.4142 #  2 #  1 # 0 #  1.4142
#         block.state_dict()[key] /= 2 #  1.4142 #  2 #  1 # 0 #  1.4142
# print(block.state_dict()['bn1.weight'])
# 
# # x0 /= 2
# x1 = block(x0)
# print(torch.norm(x1))
# # print(torch.norm(x1/2))

# bn Lipschitz test
print(block.state_dict()['bn1.weight'])
block.state_dict()['bn1.running_var'] /= 10
print(block.state_dict()['bn1.weight'])
print(torch.sqrt(block.state_dict()['bn1.running_var']))
block.eval()
for _ in range(10):
    x0 = torch.randn(10, 8, 32, 32)
    x1 = torch.randn(10, 8, 32, 32)
    print(torch.norm(block(x0) - block(x1))/torch.norm(x0 - x1))
    # deltax = torch.randn(1, 8, 32, 32)
    # print(torch.norm(block(x0 + deltax) - block(x0))/torch.norm(deltax))





