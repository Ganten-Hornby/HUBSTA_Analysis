#!/bin/bash

# Usage: bash tmux.sh wangyuhao/DMTNML/a5rv2ox9
# This script sets up a tmux session with multiple windows running specific tasks.

SESSION="dmt_nml"
ARGUMENT=$1
tmux new-session -d -s $SESSION -n "Window0"

# Activate the conda environment in the first window and run the script
tmux send-keys -t $SESSION "conda activate nml && bash run_wandb.sh $ARGUMENT 0 4" C-m

# Create other windows from 1 to 7, activate the conda environment, and run the script
for i in {1..7}; do
  tmux new-window -t $SESSION:$i -n "Window$i"
  tmux send-keys -t "Window$i" "conda activate nml && bash run_wandb.sh $ARGUMENT $i 4" C-m
done

# Switch to the first window and attach to the session
tmux select-window -t $SESSION:0
tmux -2 attach-session -t $SESSION