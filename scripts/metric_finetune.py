# To run:
# python metric.py --input_file [path_file] --metric [metric] --lm_model [model]
# input_file is path to input as CSV
# metric: options are {mask-random, mask-predict}
# model: options are {bert, roberta, albert}

import os
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import KFold, train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict


def read_data(input_file):
    """
    Load data into panda DataFrame format.
    """
    
    df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction'])
    
    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['gold-direction'] == 'stereo':
                df_item = {'sent1': row['disadvantaged'],
                           'sent2': row['advantaged']}
            else:
                df_item = {'sent1': row['advantaged'],
                           'sent2': row['disadvantaged']}
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

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)


    log_probs = torch.tensor([], requires_grad=True)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    sum_log_probs = torch.tensor(0., requires_grad=True)

    if torch.cuda.is_available():
        log_probs = log_probs.to('cuda')
        sum_log_probs = sum_log_probs.to('cuda')

    # we only need log_prob for the MASK tokens
    for i, token_id in enumerate(masked_token_ids[0]):
        if token_id.item() == mask_id:
            hs = hidden_states[i]
            target_id = token_ids[0][i]
            score = log_softmax(hs)[target_id]
            score_tensor = torch.tensor([score], requires_grad=True)
            if torch.cuda.is_available():
                score_tensor = log_probs.to('cuda')
            log_probs = torch.cat((log_probs, score_tensor), dim=0)
            sum_log_probs = torch.add(sum_log_probs, score)

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


def mask_ngram(sent1, sent2, mask_id, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    
    tokenizer = lm["tokenizer"]
    uncased = lm["uncased"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
    N = len(template1)

    # random masking
    sent1_log_probs = torch.tensor(0., requires_grad=True)
    sent2_log_probs = torch.tensor(0., requires_grad=True)
    if torch.cuda.is_available():
        sent1_log_probs = sent1_log_probs.to('cuda')
        sent2_log_probs = sent2_log_probs.to('cuda')

    total_masked_tokens = 0
    for i in range(N):
        sent1_masked_token_ids = sent1_token_ids.clone()
        sent2_masked_token_ids = sent2_token_ids.clone()

        # mask n-gram tokens
        for j in range(i, i+n):
            if j == N:
                break
            sent1_masked_token_ids[0][template1[j]] = mask_id
            sent2_masked_token_ids[0][template2[j]] = mask_id
            total_masked_tokens += 1

        _, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        _, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

        sent1_log_probs = torch.add(sent1_log_probs, score1)
        sent2_log_probs = torch.add(sent2_log_probs, score2)

    out = sent1_log_probs-sent2_log_probs
    metric_out = 0
    if out.item() > 0:
        metric_out = 1

    return out**2, metric_out


def mask_random(sent1, sent2, mask_id, lm, T=25):
    """
    Score each sentence using mask-random algorithm, following BERT masking algorithm.
    For each iteration, we randomly masked 15% of subword tokens output by model's tokenizer.
    T: number of iterations
    """


    sent1_token_ids = lm['tokenizer'].encode(sent1, return_tensors = 'pt')
    sent2_token_ids = lm['tokenizer'].encode(sent2, return_tensors = 'pt')

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
    N = len(template1)

    mask_prob = 0.15
    total_masked_tokens = 0
    
    sent1_log_probs = torch.tensor(0., requires_grad=True)
    sent2_log_probs = torch.tensor(0., requires_grad=True)
    if torch.cuda.is_available():
        sent1_log_probs = sent1_log_probs.to('cuda')
        sent2_log_probs = sent2_log_probs.to('cuda')

    num_masked_tokens = max(1, math.ceil(mask_prob * N))
    for t in range(T):
        masked_idx = np.random.choice(N, num_masked_tokens, replace=False)

        sent1_masked_token_ids = sent1_token_ids.clone()
        sent2_masked_token_ids = sent2_token_ids.clone()
        
        for idx in masked_idx:
            idx = min(len(template1)-1, idx)
            sent1_idx = template1[idx]
            sent2_idx = template2[idx]
            sent1_masked_token_ids[0][sent1_idx] = mask_id
            sent2_masked_token_ids[0][sent2_idx] = mask_id
            total_masked_tokens += 1

        _, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        _, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)


        sent1_log_probs = torch.add(sent1_log_probs, score1)
        sent2_log_probs = torch.add(sent2_log_probs, score2)

    out = sent1_log_probs-sent2_log_probs
    metric_out = 0
    if out.item() > 0:
        metric_out = 1

    return out**2, metric_out


def mask_predict(sent1, sent2, mask_id, lm, T=10):
    """
    Score each sentence using mask-predict algorithm.
    For each iteration, we unmask n words until all the words are unmasked.
    T: number of iterations
    """

    sent1_token_ids = lm['tokenizer'].encode(sent1, return_tensors = 'pt')
    sent2_token_ids = lm['tokenizer'].encode(sent2, return_tensors = 'pt')

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
    N = len(template1)

    sent1_masked_token_ids = sent1_token_ids.clone()
    sent2_masked_token_ids = sent2_token_ids.clone()

    total_unmasked_tokens = 0
    sent1_log_probs = torch.tensor(0., requires_grad=True)
    sent2_log_probs = torch.tensor(0., requires_grad=True)
    log_probs1, log_probs2 = torch.tensor([], requires_grad=True), torch.tensor([], requires_grad=True)
    if torch.cuda.is_available():
        sent1_log_probs = sent1_log_probs.to('cuda')
        sent2_log_probs = sent2_log_probs.to('cuda')
        log_probs1 = log_probs1.to('cuda')
        log_probs2 = log_probs2.to('cuda')

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
            sorted_log_probs1 = torch.sort(log_probs1, dim=0, descending=True)[1].tolist()[:num_unmasked_tokens]
            sorted_log_probs2 = torch.sort(log_probs2, dim=0, descending=True)[1].tolist()[:num_unmasked_tokens]
            # sorted_log_probs1 = sorted(log_probs1, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]
            # sorted_log_probs2 = sorted(log_probs2, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]

            # get index of the token id that has the highest log prob
            for (idx1, _), (idx2, _) in zip(sorted_log_probs1, sorted_log_probs2):
                # make sure it is masked before
                # assert sent1_masked_token_ids[0][idx1].item() == mask_id
                # assert sent2_masked_token_ids[0][idx2].item() == mask_id

                # unmask the token 
                sent1_masked_token_ids[0][idx1] = sent1_token_ids[0][idx1]
                sent2_masked_token_ids[0][idx2] = sent2_token_ids[0][idx2]

                total_unmasked_tokens += 1

        log_probs1, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        log_probs2, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

        sent1_log_probs = torch.add(sent1_log_probs, score1)
        sent2_log_probs = torch.add(sent2_log_probs, score2)

        # stop if all tokens already unmasked
        if total_unmasked_tokens == N:
            break

    out = sent1_log_probs-sent2_log_probs
    metric_out = 0
    if out.item() > 0:
        metric_out = 1

    return out**2, metric_out

def get_lm(lm_model):
    model = None
    if lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        uncased = True
    elif lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        uncased = False
    elif lm_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
        uncased = True

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)

    vocab = tokenizer.get_vocab()
    with open(lm_model + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    return {"model": model,
      "tokenizer": tokenizer,
      "mask_token": mask_token,
      "log_softmax": log_softmax,
      "uncased": uncased
    }

class my_dataset(Dataset):
    def __init__(self, sent1_list, sent2_list, mask_id_list):
        self.sent1 = sent1_list
        self.sent2 = sent2_list
        self.mask_id = mask_id_list

    def __getitem__(self, index):
        return [self.sent1[index],
                self.sent2[index],
                self.mask_id[index]]

    def __len__(self):
        return len(self.sent1)


def get_dataloader(train_df, tokenizer, uncased, mask_token):
    sent1_list = []
    sent2_list = []
    mask_id_list = []

    for index, row in train_df.iterrows():
        sent1 = row['sent1']
        sent2 = row['sent2']
        if uncased:
            sent1 = sent1.lower()
            sent2 = sent2.lower()

        mask_id = tokenizer.convert_tokens_to_ids(mask_token)

        sent1_list.append(sent1)
        sent2_list.append(sent2)
        mask_id_list.append(mask_id)

    dataset = my_dataset(sent1_list, sent2_list, mask_id_list)
    dataloader = DataLoader(
        dataset,
        batch_size=16
    )
    return dataloader

def batchloss(metric, sent1, sent2, mask_id, lm):
    sum_losses = torch.tensor(0., requires_grad=True)
    batchout = 0
    for i in range(len(sent1)):
        loss, out = metric(sent1[i], sent2[i], mask_id[i], lm)
        sum_losses = torch.add(sum_losses, loss)
        batchout += out
    return sum_losses, batchout

def print_metric_score(lm, test_dataloader, metric, desc):
    lm['model'].eval()
    out_positive = 0
    out_total = 0
    for batch in test_dataloader:
        with torch.no_grad():
            [sent1, sent2, mask_id] = batch
            loss, batchout = batchloss(metric, sent1, sent2, mask_id, lm)
            out_positive += batchout
            out_total += len(sent1)
    print('Metric score ' + desc + ': ' + str(out_positive) + ' / ' + str(out_total) + ' = ' + str(out_positive*1.0/out_total))


def evaluate(args):

    print("Input:", args.input_file)
    print("Metric:", args.metric)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    df_data = read_data(args.input_file)

    metric = mask_ngram
    if args.metric == "mask-predict":
        metric = mask_predict
    elif args.metric == "mask-random":
        metric = mask_random

    train_index_all, test_index_all, _, _ = train_test_split(range(len(df_data)), range(len(df_data)), test_size=500)
    df_train = df_data.iloc[train_index_all]
    df_test = df_data.iloc[test_index_all]

    best_epochs = 30
    best_lr = 1e-3
    best_loss = 10000000

    kf = KFold(n_splits=5, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(df_train):
        print('FOLD ' + str(fold))

        train_pairs = df_train.iloc[train_index]
        test_pairs =  df_train.iloc[test_index]

        lm = get_lm(args.lm_model)

        train_dataloader = get_dataloader(train_pairs, lm['tokenizer'], lm['uncased'], lm['mask_token'])
        test_dataloader = get_dataloader(test_pairs, lm['tokenizer'], lm['uncased'], lm['mask_token'])

        for lr in [1e-3, 1e-4, 1e-5]:
            print('lr: ' + str(lr))
            optimizer = AdamW(lm['model'].parameters(), lr=lr, eps=1e-8)
            epochs = 30
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_dataloader) * epochs)

            prev_loss = 10000000
            decrease = -1
            for epoch_i in range(0, epochs):
                print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

                lm['model'].train()
                total_train_loss = 0
                for batch in train_dataloader:
                    lm['model'].zero_grad()
                    [sent1, sent2, mask_id] = batch
                    loss, batchout = batchloss(metric, sent1, sent2, mask_id, lm)
                    total_train_loss += loss.item()
                    loss.backward()

                    # import pdb
                    # pdb.set_trace()
                    # any([p.grad != None for name, p in lm['model'].named_parameters()])
                    
                    optimizer.step()
                    scheduler.step()
                print("  Training loss: {0:.2f}".format(total_train_loss))

                with torch.no_grad():
                    lm['model'].eval()
                    total_eval_loss = 0
                    for batch in test_dataloader:
                        [sent1, sent2, mask_id] = batch
                        loss, batchout = batchloss(metric, sent1, sent2, mask_id, lm)
                        total_eval_loss += loss.item()
                    print("  Validation loss: {0:.2f}".format(total_eval_loss))

                # early stopping
                if total_eval_loss >= prev_loss:
                    decrease += 1
                else:
                    decrease = 0
                prev_loss = total_eval_loss
                if decrease >= 5:
                    break
            
            # find best hyperparameters
            if total_eval_loss < best_loss:
                best_loss = total_eval_loss
                best_epochs = epoch_i
                best_lr = lr

        fold += 1

    # finetune lm on train data with best_epochs and best_lr
    lm = get_lm(args.lm_model)
    print('Best epochs: ' + str(best_epochs))
    print('Best lr: ' + str(best_lr))
    train_dataloader = get_dataloader(df_train, lm['tokenizer'], lm['uncased'], lm['mask_token'])
    optimizer = AdamW(lm['model'].parameters(), lr=best_lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * best_epochs)
    for epoch_i in range(0, best_epochs):
        print('Epoch {:} / {:}'.format(epoch_i + 1, best_epochs))
        total_train_loss = 0
        lm['model'].train()
        for batch in train_dataloader:
            lm['model'].zero_grad()
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            [sent1, sent2, mask_id] = batch
            loss, batchout = batchloss(metric, sent1, sent2, mask_id, lm)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("  Training loss: {0:.2f}".format(total_train_loss))


    lm_unfinetuned = get_lm(args.lm_model)
    test_dataloader_unfinetuned = get_dataloader(df_test, lm_unfinetuned['tokenizer'], lm_unfinetuned['uncased'], lm_unfinetuned['mask_token'])
    print_metric_score(lm_unfinetuned, test_dataloader_unfinetuned, metric, 'before finetuning')

    test_dataloader = get_dataloader(df_test, lm['tokenizer'], lm['uncased'], lm['mask_token'])
    print_metric_score(lm, test_dataloader, metric, 'after finetuning')

    # for glue
    lm['model'].save_pretrained('finetuned_lm')
    lm['tokenizer'].save_pretrained('finetuned_lm')



parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--metric", type=str, help="metric for scoring (mask-random, mask-predict)")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use")

args = parser.parse_args()
evaluate(args)
