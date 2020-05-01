# To run:
# python compute_diff.py --input1 [path_file1] --input2 [path_file2] --output [output.csv]
# input1 and input2 should have the same number of sentences
# the format is (sent_id sentences), separated by a white space
# output_file: dump of diff in pandas DataFrame (csv format)

import os
import sys
import spacy
import argparse
import pandas as pd

from difflib import SequenceMatcher


def read_data(input_file1, input_file2):
    """
    Load data into pandas DataFrame format.
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


def get_span(seq, doc):
    spans = []
    pos_tags = []
    num_changed_words = 0

    # if the first word is changing
    if seq[0] != 0:
        curr_span = []
        curr_pos = []
        for x in range(0, seq[0], 1):
            curr_span.append(doc[x])
            curr_pos.append(doc[x].pos_)
            num_changed_words += 1
        spans.append(curr_span)
        pos_tags.append(curr_pos)

    for i in range(len(seq) - 1):
        if seq[i] + 1 != seq[i + 1]:
            curr_span = []
            curr_pos = []
            for x in range(seq[i]+1, seq[i+1], 1):
                curr_span.append(doc[x])
                curr_pos.append(doc[x].pos_)
                num_changed_words += 1
            spans.append(curr_span)
            pos_tags.append(curr_pos)
    return spans, pos_tags, num_changed_words


def filter_data(args):

    nlp = spacy.load("en_core_web_lg")

    # load data into pandas DataFrame
    df_data = read_data(args.input1, args.input2)
    df_diff = pd.DataFrame(columns=['sent1', 'sent2', 
                                    'span1', 'span2',
                                    'pos1', 'pos2',
                                    'num_changed_words1', 'num_changed_words2'])

    for index, data in df_data.iterrows():
        sent1, sent2 = data['sent1'], data['sent2']

        doc1 = nlp(sent1)
        doc2 = nlp(sent2)

        sent1 = [token.text for token in doc1]
        sent2 = [token.text for token in doc2]
        
        template1, template2 = [], []
        matcher = SequenceMatcher(None, sent1, sent2)
        for op in matcher.get_opcodes():
            if op[0] == 'equal':
                template1 += [x for x in range(op[1], op[2], 1)]
                template2 += [x for x in range(op[3], op[4], 1)]

        span1, pos1, num_changed_words1 = get_span(template1, doc1)
        span2, pos2, num_changed_words2 = get_span(template2, doc2)

        df_diff = df_diff.append({'sent1': data['sent1'],
                                  'sent2': data['sent2'],
                                  'span1': span1,
                                  'span2': span2,
                                  'pos1': pos1,
                                  'pos2': pos2,
                                  'num_changed_words1': num_changed_words1,
                                  'num_changed_words2': num_changed_words2
                                  }, ignore_index=True)

    df_diff.to_csv(args.output)


parser = argparse.ArgumentParser()
parser.add_argument("--input1", type=str, help="path to input file 1")
parser.add_argument("--input2", type=str, help="path to input file 2")
parser.add_argument("--output", type=str, help="path to csv output diff. file")

args = parser.parse_args()
filter_data(args)