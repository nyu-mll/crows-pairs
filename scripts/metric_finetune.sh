#!/bin/bash

# Make sure superglue data is in /home/rvb255/superglue_data
# You will want to replace all "/home/rvb255" with your current directory
# Run this using: sbatch --mail-user=rvb255@nyu.edu --mail-type=ALL --mem=8GB --gres=gpu:1 --time="7-0" metric_finetune.sh

module load python3/intel/3.7.3
pip install --user scikit-learn
pip install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user pandas
# pip install --user transformers
pip install --user git+https://github.com/W4ngatang/transformers@superglue
pip install --user tqdm
pip install --user tensorboardX
pip install --user -U git+http://github.com/PyTorchLightning/pytorch-lightning/
pip install --user tensorboard
pip install --user seqeval
pip install --user psutil
pip install --user sacrebleu
pip install --user rouge-score
pip install --user tensorflow_datasets

mkdir /home/rvb255/glue_out
mkdir /home/rvb255/superglue_out

python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model bert

cd transformers/examples/glue
python3 ../../utils/download_glue_data.py --data_dir /home/rvb255/glue_data
export PYTHONPATH="../":"${PYTHONPATH}"
python3 run_pl_glue.py --data_dir /home/rvb255/glue_data \
	--model_type bert \
	--task mrpc \
	--model_name_or_path home/rvb255/finetuned_lm \
	--output_dir /home/rvb255/glue_out \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--num_train_epochs 3 \
	--train_batch_size 32 \
	--seed 2 \
	--do_train \
	--do_predict
# ./run_pl.sh

cd /home/rvb255/transformers/examples
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name ax-b
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name ax-g
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name boolq
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name cb
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name copa
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name multirc
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name record
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name rte
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name wic
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type bert --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --do_lower_case --task_name wsc
# flag --do_lower_case for bert and albert

