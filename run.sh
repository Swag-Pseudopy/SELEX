#!/bin/bash

DATASET_PATH="scratch/dataset.pt"
DATASET_PATH_trial="scratch/dataset_trial.pt"

if [ -z "$DATASET_PATH" ] && [ -z "$DATASET_PATH_trial" ]; then
    echo "Usage: $0 path/to/dataset.csv path/to/dataset_trial.csv"
    exit 1
fi

# python main.py --dataset_dir "scratch/trial" --torch_save_path "$DATASET_PATH_trial"  --unique_counts_path "scratch/unique_counts_trial.pt" 2>&1 | tee "scratch/trial.log"
# python emb.py --dataset_dir "scratch/trial" --torch_save_path "$DATASET_PATH_trial"  --unique_counts_path "scratch/unique_counts_trial.pt"

# python main.py 2>&1 | tee "scratch/main.log"
# python emb.py

python main.py 2>&1 | tee "scratch/main_KL_norm.log"