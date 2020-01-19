lsof /dev/nvidia$1 | awk '{print$2}' | uniq | tail -n +2 | xargs kill -9
