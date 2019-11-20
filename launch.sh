##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

model="resnet"
depth=20 # 20 # 
grow=true
mode='adapt' # fixed
grow_atom='model' # 'layer'
err_atom='layer' # 'layer'
operation='duplicate' # plus
maxdepth=74
thresh='1.2' #'1.1'
backtrack=10
window=5

# fixed
dupEpoch=(60 110)

epochs=164
# schedule=(200 300) # dummy schedule
schedule=(81 122) # dummy schedule

gpu_id=2
log_file="train.out"
debug=0 # 

if [ "$grow" = true ]; then
    if [ "$mode" = 'fixed' ]; then
	dir="resnet-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"
    else
        # dir="resnet-$depth-"$mode-$maxdepth-"$grow_atom""wise-th=${thresh//'.'/'_'}-back=$backtrack-window=$window"
        dir="resnet-$depth-$mode-$maxdepth-$grow_atom-th=${thresh//'.'/'_'}-$err_atom-$operation"
    fi
else
    dir="resnet-$depth"
fi

if (( debug > 0 )); then
    dir="Debug-"$dir
fi

dir=$dir-"schedule"

checkpoint="checkpoints/cifar10/$dir"

[[ -f $checkpoint ]] && rm $checkpoint
i=1
while [ -d $checkpoint ]; do
    read -p "Checkpoint path $checkpoint already exists. Delete[d], Rename[r], Continue[c] or Terminate[t]? " ans
    case $ans in
	d ) rm -rf $checkpoint; break;;
	r ) checkpoint=${checkpoint%%_*}"_"$i;;
	c ) log_file="resume.out"; break;;
	* ) exit;;
    esac
    (( i++ ))
done
if [ ! -f $checkpoint ];then
    mkdir $checkpoint
fi
echo "Checkpoint path: "$checkpoint

# ./train.sh $model $depth $grow $epochs "${schedule[@]}" $gpu_id $checkpoint $debug $maxdepth $mode $thresh $backtrack $window $atom $operation $dupEpoch 2>&1 | tee $checkpoint"/"$log_file 

if [ "$grow" = true ]; then
    python cifar.py -a $model --grow --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --schedule "${schedule[@]}" --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id  --debug-batch-size $debug 2>&1 | tee $checkpoint"/"$log_file 
else
    python cifar.py -a $model --depth $depth --epochs $epochs --schedule ${schedule[@]} --gamma 0.1 --wd 1e-4 --checkpoint $checkpoint --gpu-id $gpu_id --debug-batch-size $debug | tee $checkpoint"/"$log_file 
fi

echo
echo 'Dir: '$checkpoint
echo 'copy main script to dir..'
cp cifar.py $checkpoint
