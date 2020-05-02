# To run:
# python metric.py --input1 [path_file1] --input2 [path_file2] --metric [metric] --output_file [output.csv]
# input1 and input2 should have the same number of sentences
# the format is (sent_id sentences), separated by a white space
# metric: options are {baseline, mask-random, mask-predict}
# output_file: dump of log softmax score in panda DataFrame (csv format)

import os
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from collections import defaultdict


def read_data(input_file1, input_file2):
    """
    Load data into panda DataFrame format.
    Each file should have format (separated by a white-space):
    sent_id sentence
    """
    
    df_data = pd.DataFrame(columns=['id', 'sent1', 'sent2'])
    
    data1 = [x.strip().split() for x in open(input_file1, 'r').readlines()]
    data2 = [x.strip().split() for x in open(input_file2, 'r').readlines()]

    assert len(data1) == len(data2)

    for sent1, sent2 in zip(data1, data2):
        sent_id = 'sent_' + sent1[0]

        df_item = {'id': sent_id,
                   'sent1': ' '.join(sent1[1:]),
                   'sent2': ' '.join(sent2[1:])}

        df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def compute_log_prob(masked_token_ids, token_ids, lm):
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    masked_token_ids = masked_token_ids.to('cuda')
    token_ids = token_ids.to('cuda')

    with torch.no_grad():
        # get model hidden states
        output = model(masked_token_ids)
        hidden_states = output[0].squeeze(0)

    log_probs = []
    sum_log_probs = 0.
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    for i, token_id in enumerate(masked_token_ids[0]):
        if token_id.item() == mask_id:
            hs = hidden_states[i]
            target_id = token_ids[0][i]
            log_probs.append((i, log_softmax(hs)[target_id].item()))
            sum_log_probs += log_softmax(hs)[target_id].item()

    return log_probs, sum_log_probs


def get_span(seq1, seq2):

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def baseline(data, lm):
    """
    Score sentence by masking all the words except the words that are different
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    sent1, sent2 = data["sent1"], data["sent2"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    # diff_span holds subword token ids that are different between two sentences
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    sent1_masked_token_ids = sent1_token_ids.clone().detach()
    sent2_masked_token_ids = sent2_token_ids.clone().detach()

    for idx1, idx2 in zip(template1, template2):
        sent1_masked_token_ids[0][idx1] = mask_id
        sent2_masked_token_ids[0][idx2] = mask_id

    _, sent1_log_probs = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
    _, sent2_log_probs = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

    score = {}
    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs
    score["sent1_token_score"] = sent1_log_probs / len(template1)
    score["sent2_token_score"] = sent2_log_probs / len(template2)

    return score


def mask_random(data, lm, T=5):
    """
    Score each sentence using mask-random algorithm, following BERT masking algorithm.
    For each iteration, we randomly masked 15% of subword tokens output by model's tokenizer.
    T: number of iterations
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    sent1, sent2 = data["sent1"], data["sent2"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    mask_prob = 0.15
    N = len(template1)  # num. of tokens that can be masked
    total_masked_tokens = 0
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    # random masking
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    num_masked_tokens = max(1, math.ceil(mask_prob * N))  # per iteration
    for t in range(T):
        # randomly get index to be masked
        masked_idx = np.random.choice(N, num_masked_tokens, replace=False)
        
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()
        for idx in masked_idx:
            sent1_idx = template1[idx]
            sent2_idx = template2[idx]
            sent1_masked_token_ids[0][sent1_idx] = mask_id
            sent2_masked_token_ids[0][sent2_idx] = mask_id
            total_masked_tokens += 1

        _, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        _, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)
        sent1_log_probs += score1
        sent2_log_probs += score2

    score = {}
    # average over iterations
    score["sent1_score"] = sent1_log_probs / T
    score["sent2_score"] = sent2_log_probs / T
    # average score per masked token
    score["sent1_token_score"] = sent1_log_probs / total_masked_tokens
    score["sent2_token_score"] = sent2_log_probs / total_masked_tokens

    return score


def mask_predict(data, lm, T=10):
    """
    Score each sentence using mask-predict algorithm.
    For each iteration, we unmask n words until all the words are unmasked.
    T: number of iterations
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    sent1, sent2 = data["sent1"], data["sent2"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    N = len(template1)  # num. of tokens that can be masked
    total_unmasked_tokens = 0

    sent1_masked_token_ids = sent1_token_ids.clone().detach()
    sent2_masked_token_ids = sent2_token_ids.clone().detach()

    sent1_log_probs = 0.
    sent2_log_probs = 0.
    log_probs1, log_probs2 = [], []
    for t in range(T):
        num_unmasked_tokens = int(N - (N * ((T - t) / T)))
        masked_idx = np.random.choice(N, num_unmasked_tokens, replace=False)

        if t == 0:
            # mask all tokens except the changing words
            for idx1, idx2 in zip(template1, template2):
                sent1_masked_token_ids[0][idx1] = mask_id
                sent2_masked_token_ids[0][idx2] = mask_id
        else:
            # sort log prob of tokens
            sorted_log_probs1 = sorted(log_probs1, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]
            sorted_log_probs2 = sorted(log_probs2, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]

            # get index of the token id that has the highest log prob
            for (idx1, _), (idx2, _) in zip(sorted_log_probs1, sorted_log_probs2):
                # make sure it is masked before
                assert sent1_masked_token_ids[0][idx1].item() == mask_id
                assert sent2_masked_token_ids[0][idx2].item() == mask_id

                # unmask the token 
                sent1_masked_token_ids[0][idx1] = sent1_token_ids[0][idx1]
                sent2_masked_token_ids[0][idx2] = sent2_token_ids[0][idx2]

                total_unmasked_tokens += 1

        log_probs1, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        log_probs2, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

        sent1_log_probs += score1
        sent2_log_probs += score2

        # stop if all tokens already unmasked
        if total_unmasked_tokens == N:
            break

    score = {}
    score["sent1_score"] = sent1_log_probs / (t+1)
    score["sent2_score"] = sent2_log_probs / (t+1)
    score["sent1_token_score"] = sent1_log_probs / total_unmasked_tokens
    score["sent2_token_score"] = sent2_log_probs / total_unmasked_tokens

    return score


def evaluate(args):

    print("Evaluating:")
    print("Input1:", args.input1)
    print("Input2:", args.input2)
    print("Metric:", args.metric)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame
    df_data = read_data(args.input1, args.input2)

    if args.lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        uncased = True
    elif args.lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        uncased = False
    elif args.lm_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
        uncased = True


    model.eval()
    model.to('cuda')
    # torch.set_grad_enabled(False)

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()
    with open(args.lm_model + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each sentence. 
    # each row in the dataframe has the sentid and score for pro and anti stereo.
    df_score = pd.DataFrame(columns=['id', 'sent1', 'sent2', 
                                      'sent1_score', 'sent2_score',
                                      'sent1_token_score', 'sent2_token_score',])

    metric = baseline
    if args.metric == "mask-predict":
        metric = mask_predict
    elif args.metric == "mask-random":
        metric = mask_random

    sent1, sent2, neutral = 0, 0, 0
    for index, data in df_data.iterrows():
        score = metric(data, lm)

        for stype in score.keys():
            score[stype] = round(score[stype], 3)

        best = "none"
        if score["sent1_score"] > score["sent2_score"]:
            best = "sent1"
            sent1 += 1
        elif score["sent1_score"] < score["sent2_score"]:
            best = "sent2"
            sent2 += 1
        else:
            neutral += 1

        df_score = df_score.append({'id': data['id'],
                                      'sent1': data['sent1'],
                                      'sent2': data['sent2'],
                                      'sent1_score': score['sent1_score'],
                                      'sent2_score': score['sent2_score'],
                                      'sent1_token_score': score['sent1_token_score'],
                                      'sent2_token_score': score['sent2_token_score']
                                      }, ignore_index=True)

        print(index, best, sent1, sent2, neutral)

    df_score.to_csv(args.output_file)
    print("sent1:", sent1, round(sent1 / (sent1 + sent2 + neutral) * 100, 2))
    print("sent2:", sent2, round(sent2 / (sent1 + sent2 + neutral) * 100, 2))
    print("neutral:", neutral, round(neutral / (sent1 + sent2 + neutral) * 100, 2))

    print('=' * 100)
    print()


parser = argparse.ArgumentParser()
parser.add_argument("--input1", type=str, help="path to input file 1")
parser.add_argument("--input2", type=str, help="path to input file 2")
parser.add_argument("--metric", type=str, help="metric for scoring (baseline, mask-random, mask-predict)")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use")
parser.add_argument("--output_file", type=str, help="path to output file")

args = parser.parse_args()
evaluate(args)
