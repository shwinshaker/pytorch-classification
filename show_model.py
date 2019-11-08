from models.cifar import resnet

net = resnet(num_classes=10, depth=20, block_name="basicblock")
print(net._modules.keys())
for block in net._modules['layer1']:
    print(block)
    # block.register_forward_hook(lambda x: x)
# for # layer in list(net._modules.items()):
#     # print(layer)
