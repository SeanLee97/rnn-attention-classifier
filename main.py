# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import argparse
parser = argparse.ArgumentParser(description='rnn attention')

parser.add_argument('--mode', type=str, default="train", help='train | test | valid')
parser.add_argument('--use_attention', action='store_true', default=True, help='whether to use attention')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epoch', type=int, default=50, help='batch size')
parser.add_argument('--log_step', type=int, default=2, help='batch size')

parser.add_argument('--rnn_type', type=str, default='gru', help='lstm or gru')
parser.add_argument('--bi_rnn', action='store_true', default=True, help='bidirectional')
parser.add_argument('--num_layers', type=int, default=1, help='num layers of rnn')
parser.add_argument('--embed_size', type=int, default=300, help='embed size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of rnn cell')
parser.add_argument('--attn_size', type=int, default=512, help='size of attention')
parser.add_argument('--dropout', type=float, default=0.5, help='keep prob of dropout')
parser.add_argument('--optim_type', type=str, default='sgd', help='adam | adagrad | sgd | rmsprop')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--l2_reg', type=float, default=0.001, help='l2 regularization')
parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max_grad_norm')
parser.add_argument('--runtime_dir', type=str, default='runtime', help='dir of runtime')
parser.add_argument('--ckpt_dir', type=str, default='checkpoint', help='dir of checkpoint')
parser.add_argument('--resume', action='store_true', default=False, help='resume')
parser.add_argument('--valid', action='store_true', default=True, help='is valid')

args = parser.parse_args()

import os
from dataset import load_data, Corpus, Dataset

if args.mode == 'train':
    from trainer import Trainer

    datas, labels = load_data('./corpus/TREC.train')
    corpus = Corpus(datas, labels)

    valid_datas, valid_labels = load_data('./corpus/TREC.test')
    dataset = {
        'train': Dataset(corpus, datas, labels),
        'valid': Dataset(corpus, valid_datas, valid_labels)
    }
    args.vocab_size = corpus.vocab_size
    args.label_size = corpus.label_size
    trainer = Trainer(args)
    trainer.train(dataset)