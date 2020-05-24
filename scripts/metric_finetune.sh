#!/bin/bash

module load python3/intel/3.7.3
pip install --user scikit-learn
pip install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user pandas
pip install --user transformers


# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model bert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model bert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model bert

# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model roberta
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model roberta
python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model roberta
./transformers/examples/text-classification/run_pl.sh

# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-random --lm_model albert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-predict --lm_model albert
# python metric_finetune.py --input_file data/filtered_lmBias_data.csv --metric mask-ngram --lm_model albert

# sbatch --mail-user=rvb255@nyu.edu --mail-type=ALL --mem=8GB --gres=gpu:1 --time="7-0" metric_finetune.sh