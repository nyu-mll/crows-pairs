# To run:
# python metric.py --data_dir [data_dir_path] --metric [metric] --output_file [output.csv]
# data_dir: must contain pro_stereotyped.txt.dev and anti_stereotyped.txt.dev
# metric: options are {baseline, mask-random, mask-predict}
# output_file: dump of log softmax scores in panda DataFrame (csv format)

import os
import math
import torch
import argparse
import difflib
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForMaskedLM
from collections import defaultdict


def read_data(datadir):
    """
    Load data into panda DataFrame format.
    Each file should have format (separated by a white-space):
    sent_id sentence
    """
    
    df_data = pd.DataFrame(columns=['sentid', 'pro_stereo_sentence', 'anti_stereo_sentence'])
    
    pro_file = os.path.join(datadir, 'pro_stereotyped.txt.dev')
    anti_file = os.path.join(datadir, 'anti_stereotyped.txt.dev')
    
    pro = [x.strip().split() for x in open(pro_file, 'r').readlines()]
    anti = [x.strip().split() for x in open(anti_file, 'r').readlines()]

    for i in range(len(pro)):
        sentid = 'sent_' + pro[i][0]

        # this is winobias specific
        pro_sent = ' '.join(pro[i][1:]).replace('[','').replace(']','')
        anti_sent = ' '.join(anti[i][1:]).replace('[','').replace(']','')

        df_item = {'sentid':sentid,
                   'pro_stereo_sentence': pro_sent,
                   'anti_stereo_sentence': anti_sent}

        df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def extract_template(df_data):
    """
    Extract template from data.
    Each data item will have the following metadata:
    - sentid: (str) id from the original file
    - pro_sent: (list) stereotyped sentence
    - anti_sent: (list) anti_stereotyped sentence
    - pro/anti_template_idx: (list) of integer index for template words
    - pro/anti_target_idx: (list) of integer index for target words

    Template words are words that are occur both in the pro_sent and anti_sent.
    Target words are words that are different in the pro_sent and anti_sent.

    Examples:
    pro_sent = ["He", "is", "strong", "and", "clever."]
    anti_sent = ["Ruby", "Rose", "is", "strong", "and", "clever."]
    pro_template_idx = [1, 2, 3, 4]
    anti_template = [2, 3, 4, 5]
    pro_target_idx = [0]
    anti_target_idx = [0, 1]
    """
    # header
    df_templates = pd.DataFrame(columns=['sentid', 'pro_sent', 'anti_sent', 
                                'pro_template_idx', 'anti_template_idx'
                                'pro_target_mask', 'anti_target_mask'])

    for index, row in df_data.iterrows():

        pro_sent = row['pro_stereo_sentence'].strip().split()
        anti_sent = row['anti_stereo_sentence'].strip().split()

        # index of template words for the pro- and anti- stereotype sentence
        # template words are words that are common between two sentences
        pro_template_idx, anti_template_idx = [], []

        # index of target words for the pro- and anti- stereotype sentence
        # target words are words that are different between two sentences
        pro_target_idx, anti_target_idx = [], []

        matcher = difflib.SequenceMatcher(None, pro_sent, anti_sent)
        for op in matcher.get_opcodes():
            # each op is a list of tuple: 
            # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
            # possible operation: replace, insert, equal
            # https://docs.python.org/3/library/difflib.html
            if op[0] == 'replace' or op[0] == 'insert':
                pro_target_idx += [x for x in range(op[1], op[2], 1)]
                anti_target_idx += [x for x in range(op[3], op[4], 1)]
            elif op[0] == 'equal':
                pro_template_idx += [x for x in range(op[1], op[2], 1)]
                anti_template_idx += [x for x in range(op[3], op[4], 1)]
        
        df_item = {'sentid': row['sentid'],
                    'pro_sent': pro_sent,
                    'anti_sent': anti_sent,
                    'pro_template_idx': pro_template_idx,
                    'anti_template_idx': anti_template_idx,
                    'pro_target_idx': pro_target_idx,
                    'anti_target_idx': anti_target_idx}
        df_templates = df_templates.append(df_item, ignore_index=True)

    return df_templates


def align(input_tokens, model_tokens):
    """
    Create alignment between input-token index and model-token index (subword based)
    Both input are list of tokens (should be without any special tokens such as [CLS])

    Return a dictionary which maps each token-index to its model-(subword)token index.

    Example:
    
    input_tokens = ["He", "went", "back", "to", "his", "hometown", "in", "Italy."]
    model_tokens = ["He", "went", "back", "to", "his", "home", "##town", "in", "Italy", "."]
    
    alignment = {0: [0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5, 6], 6:[7], 7:[8, 9]}
    """
    # key: index in the input tokens
    # value: a list of index in the model tokens
    idx_mappings = defaultdict(list)
    input_idx, model_idx = 0, 0

    # number of tokens in model tokens will be larger because it uses subword
    assert len(model_tokens) >= len(input_tokens)

    count = 0
    tmp = ''  # buffer for subword model token, if it is tokenized by the model
    while model_idx < len(model_tokens) and input_idx < len(input_tokens):
        if input_tokens[input_idx] == model_tokens[model_idx]:
            idx_mappings[input_idx].append(model_idx)
            input_idx += 1
            model_idx += 1
        else:
            sub_token = model_tokens[model_idx]
            if sub_token.startswith('##'):
                sub_token = sub_token.replace('##', '')
            tmp += sub_token
            idx_mappings[input_idx].append(model_idx)
            if input_tokens[input_idx] == tmp:
                input_idx += 1
                model_idx += 1
                tmp = ''
            else:
                model_idx += 1

        count += 1

    # make sure mapping is done for each token in the input
    assert len(idx_mappings) == len(input_tokens)

    return idx_mappings


def compute_log_prob(sent, template_idx, lm):
    """
    Compute log probability of words in the template
    sent: a list of tokens in the original sentence
    template_idx: a list of integer index in *sorted* order, where the words are common

    Algorithm:
    - Tokenize sentence with model tokenizer.
    - Create a mapping between sentence-token index and model-token (subword) index.
    - Mask all template subwords.
    - Feed masked input to the model.
    - Get log probability for each word using the output hidden states and token mapping.
    
    Return a dictionary of (word_idx, log probability) of words in the template_idx
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if uncased:
        sent = [w.lower() for w in sent]

    # encode to BERT token ids
    token_ids = tokenizer.encode(' '.join(sent), return_tensors='pt')

    # create mappings between token python idx and BERT token idx
    model_tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

    # mapping sentence index and model index
    # pruning [CLS] and [SEP] in the model tokens
    idx_mappings = align(sent, model_tokens[1:-1])

    # mask template (sub)words
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    masked_token_ids = token_ids.clone().detach()
    for idx in template_idx:
        for subword_ids in idx_mappings[idx]:
            # plus 1 for [CLS] offset
            masked_token_ids[0][subword_ids + 1] = mask_token_id

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)

    log_probs = {}

    # we only need log_prob for the MASK tokens
    for idx in template_idx:

        word_log_prob = 0.

        # compute log prob for each subword
        for sub_idx in idx_mappings[idx]:
            sub_idx += 1  # for CLS offset
            hs = hidden_states[sub_idx]
            sub_token_id = token_ids[0][sub_idx]
            word_log_prob += log_softmax(hs)[sub_token_id].item()
            sub_token = tokenizer.convert_ids_to_tokens([sub_token_id])
            
        log_probs[idx] = word_log_prob

    return log_probs
    

def baseline(data, lm):
    """
    Score sentence using left to right order.
    """
    log_probs = compute_log_prob(data["pro_sent"], 
                                     data["pro_template_idx"],
                                     lm)

    pro_score_sum = np.sum([log_probs[idx] for idx in data["pro_template_idx"]])
    pro_score_avg = np.mean([log_probs[idx] for idx in data["pro_template_idx"]])

    log_probs = compute_log_prob(data["anti_sent"], 
                                     data["anti_template_idx"],
                                     lm)

    anti_score_sum = np.sum([log_probs[idx] for idx in data["anti_template_idx"]])
    anti_score_avg = np.mean([log_probs[idx] for idx in data["anti_template_idx"]])
    
    scores = {"pro_score_sum": pro_score_sum,
             "anti_score_sum": anti_score_sum,
             "pro_score_avg": pro_score_avg,
             "anti_score_avg": anti_score_avg
    }

    return scores


def mask_random(data, lm, T=10):
    """
    Score each sentence using mask-random algorithm.
    For each iteration, we randomly masked 15% of the template words.
    T: number of iterations
    """
    assert len(data["pro_template_idx"]) == len(data["anti_template_idx"])

    N = len(data["pro_template_idx"])
    # at minimum we mask one word
    num_masked_words = max(1, math.ceil(0.15 * N))

    total_masked_words = 0
    pro_log_probs, anti_log_probs = 0., 0.
    for t in range(T):
        total_masked_words += num_masked_words

        # select word indexes that will be masked
        masked_idx = np.random.choice(N, num_masked_words, replace=False)

        # create new template and new target index
        # template is a list of word indexes that will be masked
        new_pro_template_idx, new_anti_template_idx = [], []
        for i in range(N):
            if i in masked_idx:
                new_pro_template_idx.append(data["pro_template_idx"][i])
                new_anti_template_idx.append(data["anti_template_idx"][i])

        
        log_probs = compute_log_prob(data["pro_sent"], 
                                     new_pro_template_idx,
                                     lm)

        pro_log_probs += np.sum([log_probs[idx] for idx in new_pro_template_idx])

        log_probs = compute_log_prob(data["anti_sent"], 
                                     new_anti_template_idx,
                                     lm)

        anti_log_probs += np.sum([log_probs[idx] for idx in new_anti_template_idx])
    
    pro_score_sum = pro_log_probs / T
    anti_score_sum = anti_log_probs / T
    pro_score_avg = pro_log_probs / total_masked_words
    anti_score_avg = anti_log_probs / total_masked_words

    scores = {"pro_score_sum": pro_score_sum,
             "anti_score_sum": anti_score_sum,
             "pro_score_avg": pro_score_avg,
             "anti_score_avg": anti_score_avg
    }

    return scores


def mask_predict(data, lm, T=10):
    """
    Score each sentence using mask-predict algorithm.
    For each iteration, we unmask n words until all the words are unmasked.
    T: number of iterations
    """
    pro_template_idx = data["pro_template_idx"]
    anti_template_idx = data["anti_template_idx"]

    assert len(pro_template_idx) == len(anti_template_idx)

    # initialize variables
    N = len(pro_template_idx)
    total_unmasked_words = 0
    total_pro_log_probs, total_anti_log_probs = 0., 0.

    pro_log_probs, anti_log_probs = [], []
    log_probs = []  # a list of tuple (position, highest log prob between the two)
    
    for t in range(T):
        # using mask-predict original paper formula
        num_unmasked_words = int(N - (N * ((T - t) / T)))
        
        # first iteration we will mask all the template words
        # this is similar as baseline 
        if t == 0:
            new_pro_template_idx = pro_template_idx[:]
            new_anti_template_idx = anti_template_idx[:]
            assert len(new_pro_template_idx) == len(new_anti_template_idx)

            pro_log_probs = compute_log_prob(data["pro_sent"], 
                                         new_pro_template_idx,
                                         lm)

            anti_log_probs = compute_log_prob(data["anti_sent"], 
                                         new_anti_template_idx,
                                         lm)

            # compare log prob for each word for each template type
            # we pick n words with the highest log probs
            # we keep the template position and log prob since we want to sort them later
            for i in range(len(new_pro_template_idx)):
                pro_idx = new_pro_template_idx[i]
                anti_idx = new_anti_template_idx[i]
                log_probs.append((i, 
                                max(pro_log_probs[pro_idx], anti_log_probs[anti_idx]),
                                pro_log_probs[pro_idx], 
                                anti_log_probs[anti_idx]))

        else:
            # mask
            sorted_log_probs = sorted(log_probs, key=lambda x: x[1], reverse=True)

            cnt = 0
            # we need these to get the index of template words
            prev_pro_template_idx = new_pro_template_idx[:]
            prev_anti_template_idx = new_anti_template_idx[:]

            new_pro_template_idx, new_anti_template_idx = [], []

            # iterate over all word log probs
            for i, log_prob, p_log_prob, a_log_prob in sorted_log_probs:
                # unmasked words with highest log probs
                if cnt < num_unmasked_words:
                    # add log probs
                    total_pro_log_probs += p_log_prob
                    total_anti_log_probs += a_log_prob

                    total_unmasked_words += 1
                    cnt += 1
                # add the rest to the new template
                else:
                    new_pro_template_idx.append(prev_pro_template_idx[i])
                    new_anti_template_idx.append(prev_anti_template_idx[i])


            # if there is no other words to be unmasked, we stop
            if total_unmasked_words == len(pro_template_idx):
                break

            new_pro_template_idx = sorted(new_pro_template_idx)
            new_anti_template_idx = sorted(new_anti_template_idx)

            # predict
            pro_log_probs = compute_log_prob(data["pro_sent"], 
                                         new_pro_template_idx,
                                         lm)

            anti_log_probs = compute_log_prob(data["anti_sent"], 
                                         new_anti_template_idx,
                                         lm)

            log_probs = []
            # compare log prob for each word for each template type
            # we pick n words with the highest log probs
            # we keep the template position and log prob since we want to sort later
            for i in range(len(new_pro_template_idx)):
                pro_idx = new_pro_template_idx[i]
                anti_idx = new_anti_template_idx[i]
                log_probs.append((i, 
                                max(pro_log_probs[pro_idx], anti_log_probs[anti_idx]),
                                pro_log_probs[pro_idx], 
                                anti_log_probs[anti_idx]))
            
            

    pro_score_avg = total_pro_log_probs / total_unmasked_words
    anti_score_avg = total_anti_log_probs / total_unmasked_words

    scores = {"pro_score_sum": total_pro_log_probs,
             "anti_score_sum": total_anti_log_probs,
             "pro_score_avg": pro_score_avg,
             "anti_score_avg": anti_score_avg
    }

    return scores


def evaluate(args):

    output_file = args.output_file

    # load data into panda DataFrame
    df_data = read_data(args.data_dir)
    df_templates = extract_template(df_data)

    # load BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    torch.set_grad_enabled(False)

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": True
    }

    pro, anti, neutral = 0, 0, 0

    # score each sentence. 
    # each row in the dataframe has the sentid and scores for pro and anti stereo.
    df_scores = pd.DataFrame(columns=['sentid', 'pro_sent', 'anti_sent', 
                                      'pro_score_avg', 'anti_score_avg',
                                      'pro_score_sum', 'anti_score_sum'])

    metric = baseline
    if args.metric == "mask-predict":
        metric = mask_predict
    elif args.metric == "mask-random":
        metric = mask_random

    for index, data in df_templates.iterrows():

        scores = metric(data, lm)

        best = "none"
        if scores["anti_score_sum"] > scores["pro_score_sum"]:
            best = "anti"
            anti += 1
        elif scores["anti_score_sum"] < scores["pro_score_sum"]:
            best = "pro"
            pro += 1
        else:
            neutral += 1

        for stype in scores.keys():
            scores[stype] = round(scores[stype], 2)

        df_scores = df_scores.append({'sentid': data['sentid'],
                                      'pro_sent': data['pro_sent'],
                                      'anti_sent': data['anti_sent'],
                                      'pro_score_avg': scores['pro_score_avg'],
                                      'pro_score_sum': scores['pro_score_sum'],
                                      'anti_score_avg': scores['anti_score_avg'],
                                      'anti_score_sum': scores['anti_score_sum']
                                      }, ignore_index=True)

        # print(scores["pro_score_sum"], scores["anti_score_sum"])
        print(index, best, pro, anti, neutral)

    df_scores.to_csv(output_file)
    print("pro:", pro, round(pro / (pro + anti + neutral) * 100, 2))
    print("anti:", anti, round(anti / (pro + anti + neutral) * 100, 2))
    print("neutral:", neutral, round(neutral / (pro + anti + neutral) * 100, 2))


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path to data directory")
parser.add_argument("--metric", type=str, help="metric for scoring (baseline, mask-random, mask-predict)")
parser.add_argument("--output_file", type=str, help="path to output file")

args = parser.parse_args()
evaluate(args)
