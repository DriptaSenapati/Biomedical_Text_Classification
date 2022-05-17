# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:48:04 2022

@author: dript
"""

import os
import json
import spacy
from tqdm import tqdm
import pandas as pd

nlp = spacy.load('en_core_web_sm')


def read_pubmed_rct(file_name, desc='reading...'):
    """
    Read Pubmed 200k RCT file and tokenize using ``spacy``
    """
    tokenized_list = []
    with open(file_name, 'r') as f:
        for line in tqdm(f.readlines(), desc=desc):
            if not line.startswith('#') and line.strip() != '':
                label, sent = line.split('\t')
                tokens = nlp(sent.strip())
                text_tokens = [token.text for token in tokens]
                pos_tokens = [token.pos_ for token in tokens]
                d = {
                    'label': label,
                    'sentence': text_tokens,
                    'pos': pos_tokens,
                    'sentence_text': sent
                }
                tokenized_list.append(d)
    return tokenized_list


def save_json_list(tokenized_list, path):
    """
    Save list of dictionary to JSON
    """
    pd.DataFrame(tokenized_list).to_csv(path,index = False)


if __name__ == '__main__':
    training_list = read_pubmed_rct(os.path.join('dataset', 'train.txt'))
    dev_list = read_pubmed_rct(os.path.join('dataset', 'dev.txt'))
    testing_list = read_pubmed_rct(os.path.join('dataset', 'test.txt'))

    save_json_list(training_list, os.path.join('dataset', 'train.csv'))
    save_json_list(dev_list, os.path.join('dataset', 'dev.csv'))
    save_json_list(testing_list, os.path.join('dataset', 'test.csv'))