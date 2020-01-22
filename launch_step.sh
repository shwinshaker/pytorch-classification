##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
debug=0 # 100 # 

model='preresnet' # "preresnet" # transresnet
dataset="cifar10" # cifar10
depth=20 # 3*2 * num_blocks_per_layer + 2
grow=true
# grow start -------------------------
mode='fixed' # 'adapt'
maxdepth=74
grow_atom='model' # 'layer'
operation='duplicate' # 'plus' # in duplicate operation the first block will be treated differently, as suggested by the baseline work
scale=false # scale the residual by stepsize? For output if not adapt
trace=('norm' 'pc2')
# ------ adapt
scale_down=True # scale the residual by activations
err_atom='model' # 'layer'
thresh='0.0' #'1.1'
backtrack=3
window=7
# ----- fixed
# dupEpoch=(60 130)
# dupEpoch=(80 130)
# dupEpoch=(10 20)
# dupEpoch=(10 30)
dupEpoch=($2 $3)
# dupEpoch=(70 110)
# dupEpoch='even' #'warm'
# grow end -------------------------

# regular hypers -----------
epochs=164 # 2 # 10
scheduler='step' # 'cosine_restart' # 'cosine' # 'acosine' # 'constant' # 'adapt' # 'constant' # 'expo' # 'cosine' # constant, step, cosine
# schedule=(81 122) 
# schedule=(54 108) # even
schedule=(60 110) # test with the same schedule
# schedule=(10 30 70 110) # test with the same schedule
# schedule=() # ($2 $3) # test with the same schedule
# schedule=(20)
regularization='' # 'truncate_error'
lr='0.1'
gamma='0.1' # 0.1 # if scheduler == step or expo
weight_decay='1e-4'
r_gamma='1e-3'
train_batch='128'
test_batch='100'

gpu_id=$1 # 4 # $3 #5
# gpu_id='1,2' # 4 # $3 #5
workers=4 # 32 # 4 * num gpus; or estimate by throughput
log_file="train.out"
suffix="" # "no_bn" # "regularization" # -res" # pca"
prefix="Batch"

if (( debug > 0 )); then
    epochs=5
    dupEpoch=(1 4) # ()
    schedule=(2 3) # (3 4) # ()
fi

if [ "$grow" = true ]; then
    if [ "$mode" = 'fixed' ]; then
	if [ "$scheduler" = 'constant' ]; then
	    dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler-"lr=${lr//'.'/'-'}"
	else
	    dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"-"lr=${lr//'.'/'-'}"
	fi
    else
        # dir="resnet-$depth-"$mode-$maxdepth-"$grow_atom""wise-th=${thresh//'.'/'_'}-back=$backtrack-window=$window"
        dir="$model-$depth-$mode-$maxdepth-$grow_atom-th=${thresh//'.'/'-'}-$err_atom-$operation-$scheduler"-"lr=${lr//'.'/'-'}"
    fi
else
    if [ "$scheduler" = 'constant' ]; then
	dir="$model-$depth-$scheduler-lr=${lr//'.'/'-'}"
    else
	dir="$model-$depth-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"-lr=${lr//'.'/'-'}"
    fi
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

if [ ! -z "$regularization" ];then
    dir=$dir-regularization
fi

if [ ! -z "$suffix" ];then
    dir=$dir-$suffix
fi

if [ ! -z "$prefix" ];then
    dir=$prefix-$dir
fi

if (( debug > 0 )); then
    dir="Debug-"$dir
fi

checkpoint="checkpoints/$dataset/$dir"
[[ -f $checkpoint ]] && rm $checkpoint
i=1
while [ -d $checkpoint ]; do
    echo '-----------------------------------------------------------------------------------------'
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
cp launch.sh $checkpoint
cp cifar.py $checkpoint
cp -r utils $checkpoint
cp -r models $checkpoint

# ./train.sh $model $depth $grow $epochs "${schedule[@]}" $gpu_id $checkpoint $debug $maxdepth $mode $thresh $backtrack $window $atom $operation $dupEpoch 2>&1 | tee $checkpoint"/"$log_file 

if [ "$grow" = true ]; then
    if (( debug > 0 )); then
	python cifar.py -d $dataset -a $model --grow --scale-stepsize $scale --scale $scale_down --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --regularization "$regularization" --r_gamma $r_gamma --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug --trace "${trace[@]}" 2>&1 | tee "$checkpoint""/"$log_file
    else
	python cifar.py -d $dataset -a $model --grow --scale-stepsize $scale --scale $scale_down --depth $depth --mode $mode --grow-atom $grow_atom --err-atom $err_atom --grow-operation $operation --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --backtrack $backtrack --window $window --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --regularization "$regularization" --r_gamma $r_gamma --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug --trace "${trace[@]}" > "$checkpoint""/"$log_file 2>&1 &
    fi
else 
    if (( debug > 0 )); then
	python cifar.py -d $dataset -a $model --scale $scale --depth $depth --epochs $epochs --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --regularization "$regularization" --r_gamma $r_gamma --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug --trace "${trace[@]}" | tee "$checkpoint""/"$log_file
    else
	python cifar.py -d $dataset -a $model --scale $scale --depth $depth --epochs $epochs --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --regularization "$regularization" --r_gamma $r_gamma --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug --trace "${trace[@]}" > "$checkpoint""/"$log_file 2>&1 &
    fi
fi
pid=$!
echo "[$pid] [$gpu_id] [Path]: $checkpoint"
if (( debug == 0 )); then
    echo "[$pid] [$gpu_id] $(date) [Path]: $checkpoint" >> log.txt
fi
