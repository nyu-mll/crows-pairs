#!/bin/bash
#SBATCH --job-name=lm-bias
#SBATCH --output=/scratch/cv50/nlu-debiasing-data/log/finetune-%j.out
#SBATCH --error=/scratch/cv50/nlu-debiasing-data/log/finetune-%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=30000
#SBATCH --signal=USR1@600
#SBATCH --mail-user=cv50@nyu.edu
#SBATCH --mail-type=END,FAIL


source activate nlu-bias

python scripts/metric_finetune.py --input_dir data/cross-val --lm_model bert --max_epochs 10 --fold 0 --lr 0.001