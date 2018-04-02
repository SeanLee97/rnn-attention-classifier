# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import RNNAttention

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

    def train(self, dataset):
       
        best_acc = 0.0

        with tf.Graph().as_default(), tf.Session() as self.session:

            xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

            with tf.variable_scope("classifier", reuse=None, initializer=xavier_initializer):
                train_obj = RNNAttention(self.args, mode='train')

            if not self.args.resume:
                self.session.run(tf.global_variables_initializer())
            else:
                ckpt = tf.train.get_checkpoint_state(self.args.ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading model from: %s" % ckpt.model_checkpoint_path)
                    tf.train.Saver().restore(self.session, ckpt.model_checkpoint_path)

            with tf.variable_scope("classifier", reuse=True, initializer=xavier_initializer):
                valid_obj = RNNAttention(self.args, mode='valid')

            train_obj.assign_lr(self.session, self.args.learning_rate)

            start_time = time.time()
            for i in range(self.args.epoch):
                if self.args.lr_decay > 0:
                    lr_decay = self.args.lr_decay ** max(i - self.args.epoch, 0.0)
                    train_obj.assign_lr(self.session, self.args.learning_rate * lr_decay)                    

                train_loss, curr_acc = self.train_epoch(self.args.batch_size, dataset['train'], train_obj.train_op, train_obj, i, shuffle=True)
                print("Epoch: %d Train loss: %.3f, Accuracy: %.4f" % (i + 1, train_loss / self.args.batch_size, curr_acc))
                
                if self.args.valid:
                    valid_loss, curr_acc = self.train_epoch(1, dataset['valid'], tf.no_op(), valid_obj, i, shuffle=False)
                    print("Epoch: %d Valid loss: %.3f, Accuracy: %.4f" % (i + 1, valid_loss / self.args.batch_size, curr_acc))
                    if curr_acc > best_acc:
                        best_acc = curr_acc
                        model_saver = tf.train.Saver()
                        model_saver.save(self.session, os.path.join(self.args.ckpt_dir, 'model.pkl'))
                        print('==== save model! ====')

                curr_time = time.time()
                print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

    def train_epoch(self, batch_size, dataset, eval_op, model_obj, epoch_num, shuffle=False):

        epoch_total_loss = 0.0
        total_correct = 0.0
        total_instances = 0.0

        for step, (input_seq_arr, length_arr, label_arr) in enumerate(dataset.next_batch(batch_size, shuffle)):

            feed_dict = {model_obj.word_input: input_seq_arr,
                         model_obj.seq_length: length_arr,
                         model_obj.label: label_arr}

            if model_obj.mode == 'train':
                loss, prediction, proba, accuracy, _ = self.session.run([model_obj.loss,
                                                                       model_obj.prediction,
                                                                       model_obj.proba,
                                                                       model_obj.accuracy,
                                                                       eval_op],
                                                                      feed_dict=feed_dict)

                total_correct += np.sum(prediction == label_arr)
                total_instances += batch_size
                epoch_total_loss += loss

            else:
                loss, prediction, proba, accuracy, _ = self.session.run([model_obj.loss,
                                                                       model_obj.prediction,
                                                                       model_obj.proba,
                                                                       model_obj.accuracy,
                                                                       eval_op],
                                                                      feed_dict=feed_dict)
                total_correct += np.sum(prediction == label_arr)
                total_instances += batch_size
                epoch_total_loss += loss

        accuracy = (total_correct / total_instances)

        return epoch_total_loss, accuracy