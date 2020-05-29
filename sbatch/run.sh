#!/bin/bash
#SBATCH --job-name=lm-bias
#SBATCH --output=/scratch/cv50/nlu-debiasing-data/log/finetune-%j.out
#SBATCH --error=/scratch/cv50/nlu-debiasing-data/log/finetune-%j.err
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=30000
#SBATCH --signal=USR1@600
#SBATCH --mail-user=cv50@nyu.edu
#SBATCH --mail-type=END,FAIL


source activate nlu-bias

# usage example for fold0 and lr=1e-3:
# sbatch sbatch/run.sh 0 1e-3
python scripts/metric_finetune.py --input_dir data/cross-val --lm_model roberta --max_epochs 10 --fold $1 --lr $2