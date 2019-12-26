##################################################
# File  Name: batch_launch.sh
#     Author: shwin
# Creat Time: Fri 13 Dec 2019 01:56:31 PM PST
##################################################

#!/bin/bash

# dup_epochs=(61 80 100 120 140 160)
# dup_epochs=(10 20 30 40 50 60 70 80)
# dup_epochs=(10 20)
# dup_epochs=(20 40 60 80 100 120)
dup_epochs=(10 30 50 70 90 110 130 150)
# dup_epochs=(40 60 80 100 120 140 160)
# dup_epochs=(40 50 60 70 80 90 100)
# dup_epochs=(70 90 110 130 150)
# gpu_ids=(7 7 7 7 7 7 7)
gpu_ids=(1 1 1 1 1 1 1 1)
for ((i=0; i<${#dup_epochs[@]}; i++));do
    ./launch.sh 1 ${dup_epochs[$i]} ${gpu_ids[$i]}
    # ./launch.sh ${dup_epochs[$i]} 90 ${gpu_ids[$i]}
done
