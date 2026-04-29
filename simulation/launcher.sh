#!/bin/bash

model_id="Qwen/Qwen3-4B-Thinking-2507"

project_dir="/home/$USER/projects/aip-rgrosse/$USER/VoteSim"

export WANDB_DISABLED=true # Optional, depending on if you want to use WandB
export HF_HOME="/scratch/$USER/hf_cache"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate

python3 -m simulation.main
