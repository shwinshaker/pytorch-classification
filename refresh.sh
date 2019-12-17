##################################################
# File  Name: refresh.sh
#     Author: shwin
# Creat Time: Sun 15 Dec 2019 11:07:00 PM PST
##################################################

#!/bin/bash

# OPTIND=1
# clear=false

while getopts "h?c" opt; do
    case "$opt" in
	h|\?)
	    echo 'help'
	    exit 0
	    ;;
	c)  clear=true
	    echo 'clear'
	    ;;
    esac
done

# shift $((OPTIND-1))

# [ "${1:-}" = "--" ] && shift

[[ $clear ]] && [[ -f log1.txt ]] && rm log1.txt

unset n
while read line
do
    echo "------------------------------------------------------"
    : $((n++))
    echo "> "$line
    path=$(echo $line | awk -F ':' '{print$NF}')
    # echo "> $path"
    tail -n 2 $path/train.out
    epochs=$(($(cat $path/log.txt | wc -l) - 1))
    echo "Epoch: $epochs"
    if [ $epochs -lt 164 ];then
	echo -e '\033[31m not finished \033[0m ' 
	[[ $clear ]] && echo "$line [$epochs]" >> log1.txt
    fi
done < log.txt

# mv log1.txt log.txt

