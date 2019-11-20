#!/bin/bash

model=$1
depth=$2
grow=$3
epochs=$4
schedule=${5[@]}
gpu_id=$6
checkpoint=$7
debug=$8
maxdepth=$9
mode=${10}
thresh=${11}
backtrack=${12}
window=${13}
atom=${14}
operation=${15}
dupEpoch=${16[@]}

if [ "$grow" = true ]; then
    python cifar.py -a $model --grow --depth $depth --mode $mode --grow-atom $atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --schedule "${schedule[@]}" --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id  --debug-batch-size $debug
else
    python cifar.py -a $model --depth $depth --epochs $epochs --schedule ${schedule[@]} --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id --debug-batch-size $debug
fi
