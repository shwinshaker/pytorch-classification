##################################################
# File  Name: launch-imagenet.sh
#     Author: shwin
# Creat Time: Tue 17 Dec 2019 11:46:52 PM PST
##################################################

#!/bin/bash

python imagenet.py -a resnet18 --data ./data --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18
