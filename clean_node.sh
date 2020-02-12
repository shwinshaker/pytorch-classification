lsof /dev/nvidia$1 | awk '{print$2}' | uniq | tail -n +2 | xargs kill -9
# cat log_metric.txt | tail -7 | awk -F '[][]' '{print$1}' | xargs kill -9
