# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import os
import re
import random
import numpy as np
from collections import defaultdict, Counter

def clean_data(string):
    string = string.lower()
    string = re.sub(r"\[", " [ ", string)
    string = re.sub(r"\]", " ] ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"-", " ", string)
    return string
    

def load_data(filename):
    datas = []
    labels = []
    all_labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            arr = line.split(' ')
            label = arr[0]
            data = ' '.join(arr[1:])
            if len(data) == 0:
                continue
            data = clean_data(data)
            sentence_list = data.split()
            datas.append(sentence_list)
            labels.append(label)
    return datas, labels
        
class Corpus(object):
    def __init__(self, datas, labels):
        flatten = lambda l: [item.lower() for sublist in l for item in sublist]
        word_count = Counter(flatten(datas)).most_common()
        self.word2index = {'<pad>': 0, '<unk>': 1}
        self.vocab_size = 2
        self.all_labels = list(set(labels))
        self.label_size = len(self.all_labels)

        for word, _ in word_count:
            self.word2index[word] = self.vocab_size
            self.vocab_size += 1
        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys())) 
        self.vocab = list(self.word2index.keys())

    def idx_sentence(self, sentence):
        inp = []
        for word in sentence:
            if word in self.word2index:
                inp.append(self.word2index[word])
            else:
                inp.append(self.word2index['<pad>'])
        return inp

    def pad_sentence(self, inp, max_len):
        inp = self.idx_sentence(inp)
        inp = inp + [self.word2index['<pad>'] for i in range(max_len-len(inp))]
        return inp[:max_len]

    def var_sentences(self, sentences, labels):
        sentences_len = np.array([len(x) for x in sentences])
        max_len = max(sentences_len)
        sentences_pad = np.array([self.pad_sentence(item, max_len) for item in sentences])
        targets = np.array([self.all_labels.index(label) for label in labels])

        return sentences_pad, sentences_len, targets

class Dataset(object):

    def __init__(self, corpus, datas, labels):
        super(Dataset, self).__init__()
        self.corpus = corpus
        self.datas = datas
        self.labels = labels
        self.dataset = list(zip(datas, labels))

    def next_batch(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self.dataset)
        sidx = 0            # start index
        eidx = batch_size    # end index
        while eidx < len(self.dataset):
            batch = self.dataset[sidx:eidx]
            batch = self._batch(batch)
            temp = eidx
            eidx = eidx + batch_size
            sidx = temp
            yield batch

        if eidx >= len(self.dataset):
            batch = self.dataset[len(self.dataset)-batch_size: ]
            batch = self._batch(batch)
            yield batch

    def _batch(self, batch):
        datas, labels = [], []
        for item in batch:
            datas.append(item[0])
            labels.append(item[1])

        input_var, input_len, target_var = self.corpus.var_sentences(datas, labels)
        return input_var, input_len, target_var