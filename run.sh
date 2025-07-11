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

# python main.py # 2>&1 | tee "scratch/main_flow.log"
# python main_flow.py # 2>&1 | tee "scratch/main_flow.log"
# python main_flow_ed_.py # 2>&1 | tee "scratch/main_flow.log"
# python main_flow_ed.py # 2>&1 | tee "scratch/main_flow.log"


pythom main.py --plot_base "scratch/plot_vMSE_res" --model_dir "scratch/model_vMSE_res" --loss "MSE" --wandb_run_name "vMSE_res" --mid_layer_abundance_injection False
pythom main.py --plot_base "scratch/plot_vMSE_res_inj" --model_dir "scratch/model_vMSE_res_inj" --loss "MSE" --wandb_run_name "vMSE_res_inj" --mid_layer_abundance_injection True
python main.py --plot_base "scratch/plot_WMSE_res" --model_dir "scratch/model_WMSE_res" --loss "WMSE" --wandb_run_name "WMSE_res"
python main.py --plot_base "scratch/plot_vKL_res" --model_dir "scratch/model_vKL_res" --loss "KL" --wandb_run_name "vKL_res"
python main_flow.py --plot_base "scratch/plot_vflow_res" --model_dir "scratch/model_vflow_res" --wandb_run_name "vflow_res"
python main_flow_ed_.py --plot_base "scratch/plot_vfmLSTM_res" --model_dir "scratch/model_vfmLSTM_res" --wandb_run_name "vfmLSTM_res"
python main_flow_ed.py --plot_base "scratch/plot_vfmTr_res" --model_dir "scratch/model_vfmTr_res" --tren 1 --wandb_run_name "vfmTr_res"
