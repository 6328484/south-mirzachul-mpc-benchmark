#!/bin/bash
# Second training batch: pure PPO (5 seeds, 1200 episodes) and pure TD3 (seeds 1-2), two lanes.
cd "$(dirname "$0")/src"
lane() { for job in "$@"; do set -- $job; python3 train.py $job 2>&1 | grep --line-buffered -E "val|Error|Trace" > ../results/logs/${1}${4}_s${2}.txt; done; }
lane "ppo 0 1200 pure" "ppo 1 1200 pure" "ppo 2 1200 pure" &
lane "ppo 3 1200 pure" "ppo 4 1200 pure" "td3 1 250 pure" "td3 2 250 pure" &
wait
echo ALL_DONE
