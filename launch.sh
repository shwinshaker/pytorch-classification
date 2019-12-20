##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
debug=0 # 

model="resnet" # transresnet
dataset="imagenet" # cifar10
# dataset="cifar10"
depth=20 # 8
grow=false # true # false # true # true # false # true # false # true
# grow start -------------------------
mode='fixed' # 'adapt'
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
# dupEpoch=(80 130)
# dupEpoch=(60 $1)
dupEpoch=($1 $2)
# dupEpoch=(70 110)
# dupEpoch='even' #'warm'
# grow end -------------------------

# regular hypers -----------
epochs=90 # 164 # 
scheduler='cosine' # 'step' # 'cosine_restart' # 'cosine' # 'acosine' # 'constant' # 'adapt' # 'constant' # 'expo' # 'cosine' # constant, step, cosine
# schedule=(81 122) 
# schedule=(54 108) # even
# schedule=(60 $1) # test with the same schedule
schedule=($1 $2) # test with the same schedule
# schedule=(10 30 70 110) # test with the same schedule
# schedule=($1 60) # test with the same schedule
# schedule=(20)
lr='0.5'
gamma='0.9' # 0.1 # if scheduler == step or expo
weight_decay='1e-4'
train_batch='256' # '128'
test_batch='200' # '100'

gpu_id='4,5,6,7' # 4 # $3 #5
# gpu_id='1,2' # 4 # $3 #5
workers=64 # 32 # 0
log_file="train.out"
suffix="-lr=${lr//'.'/'-'}"

if [ "$grow" = true ]; then
    if [ "$mode" = 'fixed' ]; then
	# dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"$suffix
	dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler$suffix
    else
        # dir="resnet-$depth-"$mode-$maxdepth-"$grow_atom""wise-th=${thresh//'.'/'_'}-back=$backtrack-window=$window"
        dir="$model-$depth-$mode-$maxdepth-$grow_atom-th=${thresh//'.'/'-'}-$err_atom-$operation-$scheduler"$suffix
    fi
else
    dir="$model-$depth-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"-lr=${lr//'.'/'-'}"
fi

### -------------------------------------------- caution!

if [ ! "$scheduler" = constant ] && [ ! "$scheduler" = cosine ] && [ ! "$scheduler" = acosine ] && [ ! "$scheduler" = cosine_restart ] ;then
    dir="$dir-gamma=${gamma//'.'/'-'}"
fi

if [ "$scale" = false ];then
    dir=$dir-"noscale"
fi

if [ "$model" = resnet ]; then
    dir="Orig-"$dir
fi

if (( debug > 0 )); then
    epochs=10
    dupEpoch=(2 4)
    schedule=(2 4)
    dir="Debug-"$dir
fi

# if [ ! -z "$suffix" ];then
#     dir=$dir-$suffix
# fi

checkpoint="checkpoints/$dataset/$dir"
[[ -f $checkpoint ]] && rm $checkpoint
i=1
while [ -d $checkpoint ]; do
    ls $checkpoint
    tail -n 5 $checkpoint/train.out
    read -p "Checkpoint path $checkpoint already exists. Delete[d], Rename[r], Continue[c] or Terminate[*]? " ans
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
    # python cifar.py -a $model --grow --scale-stepsize $scale --scale $scale_down --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --checkpoint $checkpoint --gpu-id $gpu_id  --debug-batch-size $debug 2>&1 | tee $checkpoint"/"$log_file
    python cifar.py -d $dataset -a $model --grow --scale-stepsize $scale --scale $scale_down --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint $checkpoint --gpu-id $gpu_id --workers $workers --debug-batch-size $debug > $checkpoint"/"$log_file 2>&1 &
else
    # python cifar.py -a $model --scale $scale --depth $depth --epochs $epochs --scheduler $scheduler --schedule ${schedule[@]} --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --checkpoint $checkpoint --gpu-id $gpu_id --debug-batch-size $debug | tee $checkpoint"/"$log_file
    python cifar.py -d $dataset -a $model --scale $scale --depth $depth --epochs $epochs --scheduler $scheduler --schedule ${schedule[@]} --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint $checkpoint --gpu-id $gpu_id --workers $workers --debug-batch-size $debug # > $checkpoint"/"$log_file # 2>&1 &
fi
pid=$!
echo "[$pid] [Path]: $checkpoint"
echo "[$pid] $(date) [Path]: $checkpoint" >> log.txt

