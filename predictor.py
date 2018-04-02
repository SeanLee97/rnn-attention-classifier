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

class Predictor(object):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        pass