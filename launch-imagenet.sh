##################################################
# File  Name: launch-imagenet.sh
#     Author: shwin
# Creat Time: Tue 17 Dec 2019 11:46:52 PM PST
##################################################

#!/bin/bash

python cifar.py -d 'imagenet' -a resnet18 --epochs 90 --schedule 31 61 --gamma 0.1 --train_batch 256 --test_batch 200 -c checkpoints/imagenet/resnet18
