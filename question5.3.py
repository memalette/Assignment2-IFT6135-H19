# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:54:53 2019

@author: Remi
"""

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
import math

def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)

device = torch.device("cpu")
seq_len_1= 35
seq_len_2 = 70
num_seq = 10

#Prediction for the RNN
model = RNN(emb_size=200, hidden_size=1500,
            seq_len=35, batch_size=20,
            vocab_size=10000, num_layers=2,
            dp_keep_prob=0.35).cuda()

params_path = os.path.join(os.getcwd(), 'RNN 4.1', 'best_params.pt')
model.load_state_dict(torch.load(params_path))
input = 0 * torch.ones(num_seq, dtype=torch.long).cuda()
hidden = 2.5 * torch.rand(2, num_seq, 1500).cuda()
gen_seq_35 = model.generate(input, hidden, seq_len_1).cpu().data.numpy()
gen_seq_70 = model.generate(input, hidden, seq_len_2).cpu().data.numpy()

data_path = os.path.join(os.getcwd(), 'data', 'ptb.train.txt')
word_2_id, id_2_word = _build_vocab(data_path)
print('####################################### RNN Sentences #######################################')
sentences = ['' for _ in range(num_seq)]
for i in range(num_seq):
    for j in range(seq_len_1+1):
        sentences[i] += id_2_word[gen_seq_35[j, i]] + ' '
    sentences[i] += '\n'
    print(sentences[i])
sentences = ['' for _ in range(num_seq)]
for i in range(num_seq):
    for j in range(seq_len_2+1):
        sentences[i] += id_2_word[gen_seq_70[j, i]] + ' '
    sentences[i] += '\n'
    print(sentences[i])



#Prediction for the GRU
model = GRU(emb_size=200, hidden_size=1500,
            seq_len=35, batch_size=20,
            vocab_size=10000, num_layers=2,
            dp_keep_prob=0.35).cuda()

params_path = os.path.join(os.getcwd(), 'GRU 4.1', 'best_params.pt')
model.load_state_dict(torch.load(params_path))
input = 0 * torch.ones(num_seq, dtype=torch.long).cuda()
hidden = 1.75 * torch.rand(2, num_seq, 1500).cuda()
gen_seq_35 = model.generate(input, hidden, seq_len_1).cpu().data.numpy()
gen_seq_70 = model.generate(input, hidden, seq_len_2).cpu().data.numpy()
sentences = ['' for _ in range(num_seq)]
print('####################################### GRU Sentences #######################################')
sentences = ['' for _ in range(num_seq)]
for i in range(num_seq):
    for j in range(seq_len_1+1):
        sentences[i] += id_2_word[gen_seq_35[j, i]] + ' '
    sentences[i] += '\n'
    print(sentences[i])
sentences = ['' for _ in range(num_seq)]
for i in range(num_seq):
    for j in range(seq_len_2+1):
        sentences[i] += id_2_word[gen_seq_70[j, i]] + ' '
    sentences[i] += '\n'
    print(sentences[i])
