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
# dup_epochs=(10 30 50 70 90 110 130 150)
# dup_epochs=(40 60 80 100 120 140 160)
# dup_epochs=(40 50 60 70 80 90 100)
dup_epochs=(30 50)
# gpu_ids=(7 7 7 7 7 7 7)
depth=(20 26 32 38 44 50 56 62 68 74) 
# gpu_ids=(4 5 6 7 4 5 6 7 4)
gpu_ids=(4 3)
# for ((i=0; i<${#depth[@]}; i++));do
for ((i=0; i<${#dup_epochs[@]}; i++));do
    # ./launch.sh ${gpu_ids[$i]} 10 ${dup_epochs[$i]}
    # ./launch.sh ${gpu_ids[$i]} ${depth[$i]} 
    ./launch.sh ${gpu_ids[$i]} ${dup_epochs[$i]} 70 
done
