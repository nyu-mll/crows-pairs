import os
import csv
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


def read_data(input_file):
    """
    Load data into panda DataFrame format.
    Each file should have format (separated by a white-space):
    sent_id sentence
    """
    
    df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias'])

    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_item = {'sent1': row['disadvantaged'],
                       'sent2': row['advantaged'],
                       'direction': row['gold-direction'],
                       'bias': row['gold-bias']}
            df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def compute_log_prob(masked_token_ids, token_ids, lm):
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
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


def mask_random(data, lm, T=10):
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
    # score["sent1_token_score"] = sent1_log_probs / total_masked_tokens
    # score["sent2_token_score"] = sent2_log_probs / total_masked_tokens

    return score


def mask_ngram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
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

    N = len(template1)  # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    # random masking
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0
    for i in range(N):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()

        # mask n-gram tokens
        for j in range(i, i+n):
            if j == N:
                break
            sent1_masked_token_ids[0][template1[j]] = mask_id
            sent2_masked_token_ids[0][template2[j]] = mask_id
            total_masked_tokens += 1

        _, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        _, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

        sent1_log_probs += score1
        sent2_log_probs += score2

    score = {}
    # average over iterations
    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs
    # average score per masked token
    # score["sent1_token_score"] = sent1_log_probs / total_masked_tokens
    # score["sent2_token_score"] = sent2_log_probs / total_masked_tokens

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
    # score["sent1_token_score"] = sent1_log_probs / total_unmasked_tokens
    # score["sent2_token_score"] = sent2_log_probs / total_unmasked_tokens

    return score


def evaluate(args):

    print("Evaluating:")
    print("Input:", args.input_file)
    print("Metric:", args.metric)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame
    df_data = read_data(args.input_file)

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
    if torch.cuda.is_available():
        model.to('cuda')

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
    df_score = pd.DataFrame(columns=['sent1', 'sent2', 
                                     'sent1_score', 'sent2_score',
                                     'score', 'gold-direction', 'gold-bias'])

    metric = baseline
    if args.metric == "mask-predict":
        metric = mask_predict
    elif args.metric == "mask-random":
        metric = mask_random
    elif args.metric == "mask-ngram":
        metric = mask_ngram


    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0

    N = 0
    neutral = 0
    for index, data in df_data.iterrows():
        direction = data['direction']
        bias = data['bias']
        score = metric(data, lm)

        for stype in score.keys():
            score[stype] = round(score[stype], 3)

        N += 1
        pair_score = 0
        if score['sent1_score'] == score['sent2_score']:
            neutral += 1
        else:
            if direction == 'stereo':
                total_stereo += 1
                if score['sent1_score'] > score['sent2_score']:
                    stereo_score += 1
                    pair_score = 1
            elif direction == 'antistereo':
                total_antistereo += 1
                if score['sent2_score'] > score['sent1_score']:
                    antistereo_score += 1
                    pair_score = 1

        df_score = df_score.append({'sent1': data['sent1'],
                                    'sent2': data['sent2'],
                                    'sent1_score': score['sent1_score'],
                                    'sent2_score': score['sent2_score'],
                                    'score': pair_score,
                                    'gold-direction': direction,
                                    'gold-bias': bias
                                  }, ignore_index=True)

        print(index, stereo_score + antistereo_score, df_data.shape[0])

    df_score.to_csv(args.output_file)
    print('Total examples:', N)
    print('Metric score:', round((stereo_score + antistereo_score) / N * 100, 2))
    print('Stereotype score:', round(stereo_score  / total_stereo * 100, 2))
    if antistereo_score != 0:
        print('Anti-stereotype score:', round(antistereo_score  / total_antistereo * 100, 2))
    print("Num. neutral:", neutral, round(neutral / N * 100, 2))

    print('=' * 100)
    print()


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--metric", type=str, help="metric for scoring (baseline, mask-random, mask-predict, mask-ngram)")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use")
parser.add_argument("--output_file", type=str, help="path to output file")

args = parser.parse_args()
evaluate(args)
