# CrowS-Pairs

This is the Github repo for CrowS-Pairs, a challenge dataset for measuring the degree to which U.S. stereotypical biases present in the masked language models (MLMs). The associated paper is to be published as part of The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020).

## The Dataset

The dataset along with its annotations is in [crows_pairs_anonymized.csv](https://github.com/nyu-mll/nlu-debiasing-data/blob/public/data/crows_pairs_anonymized.csv). It consists of 1,508 examples covering nine types of biases: race/color, gender/gender identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status.

Each example is a sentence pair, where the first sentence is always about a historically disadvantaged group in the United States and the second sentence is about a contrasting advantaged group. The first sentence can _demonstrate_ or _violate_ a stereotype. The other sentence is a minimal edit of the first sentence: The only words that change between them are those that identify the group. Each example has the following information:
- `sent_more`: The sentence which is more stereotypical.
- `sent_less`: The sentence which is less stereotypical.
- `stereo_antistereo`: The stereotypical direction of the pair. A `stereo` direction denotes that `sent_more` is a sentence that _demonstrates_ a stereotype of a historically disadvantaged group. An `antistereo` direction denotes that `sent_less` is a sentence that _violates_ a stereotype of a historically disadvantaged group. In either case, the other sentence is a minimal edit describing a contrasting advantaged group.
- `bias_type`: The type of biases present in the example.
- `annotations`: The annotations of bias types from crowdworkers.
- `anon_writer`: The _anonymized_ id of the writer.
- `anon_annotators`: The _anonymized_ ids of the annotators.

## Quantifying Stereotypical Biases in MLMs

### Requirement

Use Python 3 (we use Python 3.7) and install the required packages.

```
pip install -r requirements.txt
```

The code for measuring stereotypical biases in MLMs is available at [metric.py](https://github.com/nyu-mll/nlu-debiasing-data/blob/public/scripts/metric.py). You can run the code using the following command:
```
python scripts/metric.py 
	--input_file data/crows_pairs_anonymized.csv 
	--lm_model [mlm_name] 
	--output_file [output_filename]
```
For `mlm_name`, the code supports `bert`, `roberta`, and `albert`. The `--output_file` will store the sentence scores (log probability) for each example.


## Data Statements

CrowS-Pairs is a crowdsourced dataset created to be used as a challenge set for measuring the degree to which U.S. stereotypical biases are present in large pretrained masked language models such as [BERT](https://www.aclweb.org/anthology/N19-1423/). The dataset consists of 1,508 examples that cover stereotypes dealing with nine type of social bias. 

Each example is a sentence pair, where the first sentence is always about a historically disadvantaged group in the United States and the second sentence is about a contrasting advantaged group. The first sentence can _demonstrate_ or _violate_ a stereotype. The other sentence is a minimal edit of the first sentence: The only words that change between them are those that identify the group. 

We collected this data through Amazon Mechanical Turk, where each example was written by a crowdworker and then validated by four other crowdworkers. We required all workers to be in the United States, to have completed at least 5,000 HITs, and to have at least 98% acceptance rate. Workers were paid a $15 hourly wage.

CrowS-Pairs covers a broad range of bias types: race, gender/gender identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status. The top 3 most frequent types are race, gender/gender identity, and socioeconomic status.

The data presented in this paper is of a sensitive nature. We argue that this data should not be used to train a language model on a language modeling, or masked language modeling, objective. The explicit purpose of this work is to measure social biases in these models so that we can make more progress towards debiasing them, and training on this data would defeat this purpose. 

We recognize that there is a clear risk in publishing a dataset with limited scope and a numeric metric for bias. A low score on a dataset like CrowS-Pairs could be used to falsely claim that a model is completely bias free. We strongly caution against this. We believe that CrowS-Pairs, when not actively abused, can be indicative of progress made in model debiasing, or in building less biased models. It is not, however, an assurance that a model is truly unbiased.






