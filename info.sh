##################################################
# File  Name: batch_move.sh
#     Author: shwin
# Creat Time: Sun 15 Dec 2019 01:28:08 PM PST
##################################################

#!/bin/bash

# dup_epochs=(10 20 30 40 50)
dup_epochs=(110 120 130 140 150 160)
for ((i=0; i<${#dup_epochs[@]}; i++));do
    # dest="checkpoints/cifar10/Orig-resnet-20-fixed-${dup_epochs[$i]}-110-duplicate-cosine-lr=0.5-noscale"
    dest="checkpoints/cifar10/Orig-resnet-20-fixed-100-${dup_epochs[$i]}-duplicate-constant-lr=0-1-noscale"
    # dest="checkpoints/cifar10/Orig-resnet-74-cosine-${dup_epochs[$i]}-60-lr=0-2-noscale"
    echo '> '$dest
    # ls $desta
    cat $dest/train.out | tail -5
    # cat $dest/train.out | grep -i 'learning rate'
    new_dest="checkpoints/cifar10/Orig-resnet-20-fixed-100-${dup_epochs[$i]}-duplicate-constant-noscale"
    mv $dest $new_dest
done
