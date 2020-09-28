# nlu-debiasing-data
This is the Github repo for CrowS-Pairs, NYU's language model bias measurement project.

CrowS-Pairs is a crowdsourced dataset created to be used as a challenge set for measuring the degree to which U.S. stereotypical biases are present in large pretrained masked language models such as [BERT](https://www.aclweb.org/anthology/N19-1423/). The dataset consists of 1,508 examples that cover stereotypes dealing with nine type of social bias. Each example is a sentence pair, where one sentence is always more stereotypical -- either demonstrating a stereotype or violating a stereotype -- than the other sentence. The more stereotypical sentence is always about a historically disadvantaged group in the United States, and the paired sentence is about a contrasting advantaged group. The two sentences are minimally distant; the only words that change between them are those that identify the group. We use this format so that we can quantify the degree to which language models prefer one sentence over the other.

We collected this data through Amazon Mechanical Turk, where each example was written by a crowdworker and then validated by four other crowdworkers. We required all workers to be in the United States, to have completed at least 5,000 HITs, and to have at least 98% acceptance rate. Workers were paid a $15 hourly wage.

CrowS-Pairs covers a broad range of bias types: race, gender/gender identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status. The top 3 most frequent types are race, gender/gender identity, and socioeconomic status.

The data presented in this paper is of a sensitive nature. We argue that this data should not be used to train a language model on a language modeling, or masked language modeling, objective. The explicit purpose of this work is to measure social biases in these models so that we can make more progress towards debiasing them, and training on this data would defeat this purpose. 

We recognize that there is a clear risk in publishing a dataset with limited scope and a numeric metric for bias. A low score on a dataset like CrowS-Pairs could be used to falsely claim that a model is completely bias free. We strongly caution against this. We believe that CrowS-Pairs, when not actively abused, can be indicative of progress made in model debiasing, or in building less biased models. It is not, however, an assurance that a model is truly unbiased.

The dataset is in [filtered_lmBias_data_clean.csv](https://github.com/nyu-mll/nlu-debiasing-data/blob/public/data/filtered_lmBias_data_clean.csv)

The full set of annotations is in [crows_pairs_anonymized.csv](https://github.com/nyu-mll/nlu-debiasing-data/blob/public/data/crows_pairs_anonymized.csv)

The associated paper is to be published as part of The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020).
