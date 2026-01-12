#!/bin/bash
# Usage:
#   chmod +x train_evaluate.sh
#   ./run_exp.sh EXP_NAME CUDA_DEVICE
# Example:
#   ./run_exp.sh NEXT_mamba1_vs_mamba2 1
# Alias:
#   echo 'runexp() { nohup ./train_evaluate.sh "$1" "$2" >/dev/null 2>&1 & }' >> ~/.bashrc
#   source ~/.bashrc
#   runexp NEXT_mamba2_more_features 1


set -e

EXP_NAME="${1:?Missing EXP_NAME}"
CUDA_DEVICE="${2:?Missing CUDA_DEVICE}"

mkdir -p logs logs/pids

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python train_multiple.py --exp_name "$EXP_NAME" \
  > logs/$EXP_NAME.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > logs/pids/${EXP_NAME}_train.pid

wait $TRAIN_PID

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python evaluate_multiple.py --exp_name "$EXP_NAME" \
  >> logs/$EXP_NAME.log 2>&1 &
EVAL_PID=$!
echo $EVAL_PID > logs/pids/${EXP_NAME}_eval.pid
