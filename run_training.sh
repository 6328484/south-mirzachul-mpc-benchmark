#!/bin/bash
# Trains all agents on two CPU cores (two lanes). Logs: results/logs/*.txt
cd "$(dirname "$0")/src"
rm -f ../results/policies/*.pkl ../results/logs/*.json ../results/logs/*.txt
lane() { for job in "$@"; do set -- $job; python3 train.py $job 2>&1 | grep --line-buffered -E "val|Error|Trace" > ../results/logs/${1}${4}_s${2}.txt; done; }
lane "ppo 0 480" "ppo 1 480" "ppo 2 480" "td3 3 250" "td3 4 250" "ppo 0 480 pure" &
lane "td3 0 250" "td3 1 250" "td3 2 250" "ppo 3 480" "ppo 4 480" "td3 0 250 pure" &
wait
echo ALL_DONE
