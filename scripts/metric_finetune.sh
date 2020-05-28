#!/bin/bash

module load python3/intel/3.7.3
pip install --user scikit-learn
pip install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user pandas
# pip install --user transformers
pip install --user git+https://github.com/W4ngatang/transformers@superglue
pip install --user tqdm

# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model bert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model bert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model bert

# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model roberta
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model roberta
python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model roberta
cd transformers/examples/glue
./run_pl.sh
cd /home/rvb255/transformers/examples
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name ax-b
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name ax-g
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name boolq
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name cb
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name copa
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name multirc
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name record
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name rte
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name wic
python run_superglue.py --data_dir /home/rvb255/superglue_data --model_type roberta --model_name_or_path /home/rvb255/finetuned_lm --output_dir /home/rvb255/superglue_out --task_name wsc
# flag --do_lower_case for bert and albert

# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model albert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model albert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model albert


# sbatch --mail-user=rvb255@nyu.edu --mail-type=ALL --mem=8GB --gres=gpu:1 --time="7-0" metric_finetune.sh
# srun --gres=gpu:1 --mem=8GB --pty /bin/bash