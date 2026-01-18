# chmod +x robust.sh
# nohup ./robust.sh > logs/robust.log 2>&1 & 1320602

#!/bin/bash
for i in {1..5}; do
  CUDA_VISIBLE_DEVICES=2 python mycode/denoising/evaluate_downstream.py \
    --config mycode/denoising/configs/all_100_no_bandpass_$i.yaml \
    --classification-fs 100 --classifiers all
done
