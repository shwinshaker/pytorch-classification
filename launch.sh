##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
debug=0 # 

model="resnet" # transresnet
depth=68
grow=false # true # true # true # false # true # false # true
# grow start -------------------------
mode='adapt' # 'adapt'
maxdepth=74
grow_atom='model' # 'layer'
operation='duplicate' # 'plus' # in duplicate operation the first block will be treated differently, as suggested by the baseline work
scale=false # scale the residual by stepsize? For output if not adapt
# ------ adapt
scale_down=True # scale the residual by activations
err_atom='model' # 'layer'
thresh='0.0' #'1.1'
backtrack=3
window=7
# ----- fixed
# dupEpoch=(60 130)
dupEpoch=(80 130)
# dupEpoch='even' #'warm'
# grow end -------------------------

# regular hypers -----------
epochs=164
scheduler='constant' # 'adapt' # 'constant' # 'expo' # 'cosine' # constant, step, cosine
# schedule=(81 122) 
schedule=(60 110) # test with the same schedule
lr='0.1'
gamma='0.9' # 0.1 # if scheduler == step or expo
weight_decay='1e-4'
train_batch='128'

gpu_id=2
log_file="train.out"

if [ "$grow" = true ]; then
    if [ "$mode" = 'fixed' ]; then
	dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler
    else
        # dir="resnet-$depth-"$mode-$maxdepth-"$grow_atom""wise-th=${thresh//'.'/'_'}-back=$backtrack-window=$window"
        dir="$model-$depth-$mode-$maxdepth-$grow_atom-th=${thresh//'.'/'-'}-$err_atom-$operation-$scheduler"
    fi
else
    dir="$model-$depth"
fi

### -------------------------------------------- caution!

if [ ! "$scheduler" = constant ] && [ ! "$scheduler" = cosine ];then
    dir="$dir-gamma=${gamma//'.'/'-'}"
fi

if [ "$scale" = false ];then
    dir=$dir-"noscale"
fi

if [ "$model" = resnet ]; then
    dir="Orig-"$dir
fi

if (( debug > 0 )); then
    # epochs=20
    dupEpoch=(2 4)
    schedule=(3 7)
    dir="Debug-"$dir
fi

# if [ ! -z "$suffix" ];then
#     dir=$dir-$suffix
# fi

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
echo 'Save main script to dir..'
cp cifar.py $checkpoint
cp -r utils $checkpoint
cp -r models $checkpoint

# ./train.sh $model $depth $grow $epochs "${schedule[@]}" $gpu_id $checkpoint $debug $maxdepth $mode $thresh $backtrack $window $atom $operation $dupEpoch 2>&1 | tee $checkpoint"/"$log_file 

if [ "$grow" = true ]; then
    python cifar.py -a $model --grow --scale-stepsize $scale --scale $scale_down --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --checkpoint $checkpoint --gpu-id $gpu_id  --debug-batch-size $debug 2>&1 | tee $checkpoint"/"$log_file 
else
    python cifar.py -a $model --scale $scale --depth $depth --epochs $epochs --scheduler $scheduler --schedule ${schedule[@]} --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --checkpoint $checkpoint --gpu-id $gpu_id --debug-batch-size $debug | tee $checkpoint"/"$log_file 
fi

echo "Checkpoint path: "$checkpoint
