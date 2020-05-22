#!/bin/bash

module load python3/intel/3.7.3
pip install --user scikit-learn
pip install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user pandas
pip install --user transformers


python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-random --lm_model bert
python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-predict --lm_model bert

python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-random --lm_model roberta
python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-predict --lm_model roberta

python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-random --lm_model albert
python metric_finetune.py --input1 data/bibimbap/pro_stereotyped.txt.dev --input2 data/bibimbap/anti_stereotyped.txt.dev --metric mask-predict --lm_model albert

# sbatch --mail-user=rvb255@nyu.edu --mail-type=ALL --mem=8GB --gres=gpu:1 --time="7-0" metric_finetune.sh

# python metric_finetune.py --input1 data/banana_advantage/advantage.txt --input2 data/banana_advantage/banana.txt --metric mask-random --lm_model bert