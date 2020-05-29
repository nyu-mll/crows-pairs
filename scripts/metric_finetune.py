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

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict


def load_data(input_file, uncased):
    """
    Load data into panda DataFrame format.
    """
    
    df_data = pd.DataFrame(columns=['sent1', 'sent2'])
    
    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent1, sent2 = row['sent1'], row['sent2']
            if uncased:
                sent1 = sent1.lower()
                sent2 = sent2.lower()
            df_item = {'sent1': sent1,
                       'sent2': sent2}
            df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def compute_log_prob(masked_token_ids, token_ids, mask_idx, lm):
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


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


def mask_ngram(sent1, sent2, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    
    tokenizer = lm["tokenizer"]
    uncased = lm["uncased"]
    mask_token = lm["mask_token"]
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

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

    total_masked_tokens = 0
    # we skip mask for [CSL] and [SEP]
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone()
        sent2_masked_token_ids = sent2_token_ids.clone()

        # mask n-gram tokens
        for j in range(i, i+n):
            if j == N:
                break
            sent1_masked_token_ids[0][template1[j]] = mask_id
            sent2_masked_token_ids[0][template2[j]] = mask_id
            total_masked_tokens += 1

        score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, template1[j], lm)
        score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, template2[j], lm)

        sent1_log_probs = torch.add(sent1_log_probs, score1)
        sent2_log_probs = torch.add(sent2_log_probs, score2)

    loss = (sent1_log_probs - sent2_log_probs).pow(2)

    return loss


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


class debias_dataset(Dataset):
    def __init__(self, sent1_list, sent2_list):
        self.sent1 = sent1_list
        self.sent2 = sent2_list

    def __getitem__(self, index):
        return [self.sent1[index],
                self.sent2[index]]

    def __len__(self):
        return len(self.sent1)


def get_dataloader(train_df, batch_size):
    dataset = debias_dataset(train_df['sent1'].tolist(), train_df['sent2'].tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )
    return dataloader


def compute_loss(sent1, sent2, lm, bs):
    total_loss = torch.tensor(0., requires_grad=True)
    assert len(sent1) == len(sent2)
   
    for i in range(len(sent1)):
        loss = mask_ngram(sent1[i], sent2[i], lm)
        total_loss = torch.add(total_loss, loss)

    return total_loss / bs


def get_initial_loss(dataloader, lm, bs):

    model = lm['model']
    total_eval_loss = 0
    with torch.no_grad():
        model.eval()
        for ibatch, batch in enumerate(dataloader):
            sent1, sent2 = batch
            eval_loss = compute_loss(sent1, sent2, lm, bs)
            total_eval_loss += eval_loss.item()

    return total_eval_loss


def fine_tune(args):

    model_dir = 'model_' + str(args.fold) + '_' + str(args.lr)
    if os.path.exists(model_dir):
        os.rmdir(model_dir)
    os.mkdir(model_dir)

    log = open(os.path.join(model_dir, 'log.log'), 'w')

    log.write('Configurations:\n')
    for k, v in vars(args).items():
        log.write(str(k) + ': ' + str(v) + '\n')
        print(k, v)

    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    lr = args.lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    lm = get_lm(args.lm_model)
    
    train_file = os.path.join(args.input_dir, 'fold' + str(args.fold) + 'train.csv')
    val_file = os.path.join(args.input_dir, 'fold' + str(args.fold) + 'val.csv')

    df_train = load_data(train_file, lm['uncased'])
    df_val = load_data(val_file, lm['uncased'])

    train_dataloader = get_dataloader(df_train, batch_size)
    val_dataloader = get_dataloader(df_val, batch_size)

    model = lm['model']
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    num_steps = len(train_dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    initial_loss = get_initial_loss(val_dataloader, lm, batch_size)
    print('Initial validation loss: ' + str(initial_loss) + '\n')
    log.write('Initial validation loss: ' + str(initial_loss) + '\n')

    prev_loss = 10000000
    best_loss = 10000000
    best_epoch = 0
    decrease = -1
    for epoch_i in range(0, max_epochs):
        print('Epoch {:} / {:}'.format(epoch_i + 1, max_epochs))
        log.write('Epoch {:} / {:}\n'.format(epoch_i + 1, max_epochs))
        total_train_loss, total_eval_loss = 0, 0
        model.train()
        for ibatch, train_batch in enumerate(train_dataloader):
            model.zero_grad()
            sent1, sent2 = train_batch
            train_loss = compute_loss(sent1, sent2, lm, batch_size)
            total_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            log.write("  Training batch {:d} loss: {:0.2f}\n".format(ibatch, train_loss.item()))
            print("  Training batch {:d} loss: {:0.2f}".format(ibatch, train_loss.item()))
            log.flush()

        with torch.no_grad():
            model.eval()
            total_eval_loss = 0
            for ibatch, eval_batch in enumerate(val_dataloader):
                sent1, sent2 = eval_batch
                eval_loss = compute_loss(sent1, sent2, lm, batch_size)
                total_eval_loss += eval_loss.item()
                log.write("  Validation batch {:d} loss: {:0.2f}\n".format(ibatch, eval_loss.item()))
                print("  Validation batch {:d} loss: {:0.2f}".format(ibatch, eval_loss.item()))
                log.flush()
        
        log.write("Total Training loss: {:0.2f}\n".format(total_train_loss))
        log.write("Total validation loss: {:0.2f}\n".format(total_eval_loss))
        print("Total Training loss: {:0.2f}".format(total_train_loss))
        print("Total validation loss: {:0.2f}".format(total_eval_loss))
        log.flush()

        # early stopping
        if total_eval_loss >= prev_loss:
            decrease += 1
        else:
            decrease = 0

        prev_loss = total_eval_loss
        if best_loss > total_eval_loss:
            best_loss = total_eval_loss
            best_epoch = epoch_i
            model.save_pretrained(model_dir)

        if decrease >= 5:
            log.write("Validation loss didn't go down for 5 epochs, stopping training early.\n")
            log.write('Best epoch:' + str(best_epoch) + '\n')
            log.write('Best validation loss' + str(best_loss) + '\n')
            log.close()
            break

    log.write('Best epoch:' + str(best_epoch) + '\n')
    log.write('Best validation loss' + str(best_loss) + '\n')
    print('Best epoch:' + str(best_epoch))
    print('Best validation loss' + str(best_loss))
    log.close()

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="path to input file")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use")
parser.add_argument("--fold", type=int, default=0, help="validation split fold (0, 1, 2, 3, 4)")
# hyperparameters
parser.add_argument("--lr", type=float, default=1e-5, help="learning_rate")
parser.add_argument("--max_epochs", type=int, default=1, help="maximum number of epochs to run")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")

args = parser.parse_args()
fine_tune(args)
