##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

model="resnet"
depth=20
grow=true
epochs=164 # 100
schedule=(200 300) # fix this later
gpu_id=1
log_file="train.out"

if [ "$grow" = true ]; then
    checkpoint="checkpoints/cifar10/resnet-"$depth"-grow"
else
    checkpoint="checkpoints/cifar10/resnet-"$depth
fi

[[ -f $checkpoint ]] && rm $checkpoint
i=1
while [ -d $checkpoint ]; do
    read -p "Checkpoint path $checkpoint already exists. Delete[d], Rename[r], Continue[c] or Terminate[t]? " ans
    case $ans in
	d ) rm -rf $checkpoint; break;;
	r ) checkpoint=$checkpoint"_"$i;;
	c ) log_file="resume.out"; break;;
	* ) exit;;
    esac
    (( i++ ))
done
if [ ! -f $checkpoint ];then
    mkdir $checkpoint
fi
echo "Checkpoint path: "$checkpoint

./train.sh $model $depth $grow $epochs $schedule $gpu_id $checkpoint 2>&1 | tee $checkpoint"/"$log_file 
