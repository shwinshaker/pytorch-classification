#!/bin/bash

model=$1
depth=$2
grow=$3
epochs=$4
schedule=$5
gpu_id=$6
checkpoint=$7

if [ "$grow" = true ]; then
    python cifar.py -a $model --grow --depth $depth --epochs $epochs --schedule $schedule --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id 
else
    python cifar.py -a $model --depth $depth --epochs $epochs --schedule $schedule --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id 
fi
