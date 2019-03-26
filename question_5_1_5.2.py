# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:55:22 2019

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
import matplotlib.pyplot as plt
# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU 
from models import make_model as TRANSFORMER
from ptblm import run_epoch2, ptb_raw_data, run_epoch, run_epoch3


# LOAD DATA

#print('Loading data from '+args.data)
raw_data = ptb_raw_data(data_path=os.path.join(os.getcwd(),'data'))
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

###########################################################

#                            5.1                              

###########################################################


############################# Model RNN ############################

# Path of the folder where the model paramters are saved
path = os.path.join(os.getcwd(),'best_model','RNN 4.1', 'best_params.pt')

# Initializing the model
model = RNN(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2, 
                dp_keep_prob=0.35).cuda()

#Setting the parameters to the one ontained in 4.1
model.load_state_dict(torch.load(path))

# Calculating the loss
val_loss,k = run_epoch2(model, valid_data)

# Calculation the average loss for all the time step
val = val_loss.cpu().numpy()/k
val = val.reshape((35,20))
out_RNN=val.mean(1)

plt.plot(out_RNN)


############################# Model GRU ############################

# Path of the folder where the model paramters are saved
path = os.path.join(os.getcwd(),'best_model','GRU 4.1', 'best_params.pt')

# Initializing the model
model = GRU(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2, 
                dp_keep_prob=0.35).cuda()

#Setting the parameters to the one ontained in 4.1
model.load_state_dict(torch.load(path))

# Calculating the loss
val_loss,k = run_epoch2(model, valid_data)

# Calculation the average loss for all the time step
val = val_loss.cpu().numpy()/k
val = val.reshape((35,20))
out_GRU = val.mean(1)

plt.plot(out_GRU)


############################# Model TRANSFORMER ############################

# Path of the folder where the model paramters are saved
path = os.path.join(os.getcwd(),'best_model','TRANSFORMER 4.1', 'best_params.pt')

# Initializing the model
model = TRANSFORMER(vocab_size=10000, n_units=512, 
                            n_blocks=6, dropout=0.1)
model.batch_size=20
model.seq_len=35
model.vocab_size=10000

#Setting the parameters to the one ontained in 4.1
model.load_state_dict(torch.load(path))

# Calculating the loss
val_loss,k = run_epoch3(model, valid_data,'TRANSFORMER')

# Calculation the average loss for all the time step
val = val_loss.cpu().numpy()/k
val = val.reshape((35,20))
out_TRNSFMR = val.mean(1)

plt.plot(out_TRNSFMR)



plt.plot(out_GRU)
plt.plot(out_RNN)
plt.plot(out_TRNSFMR)
plt.show()   


###########################################################

#                            5.2                              

###########################################################

############################# Model RNN ############################

# Path of the folder where the model paramters are saved
path = os.path.join(os.getcwd(),'best_model','RNN 4.1', 'best_params.pt')

# Initializing the model
model = RNN(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2, 
                dp_keep_prob=0.35).cuda()

#Setting the parameters to the one ontained in 4.1
model.load_state_dict(torch.load(path))

# Calculating the loss
run_epoch(model, valid_data, is_train=True, lr=1.0,grad_show=True)


h1_RNN=[]
for i in range(35):
    h1_RNN.append(torch.norm(torch.stack(model.gradients[2*i:2*i+1])))
    
h_RNN=torch.stack(h1_RNN) 
h_RNN=h_RNN.cpu().numpy()
h_RNN=(h_RNN-np.min(h_RNN))/(np.max(h_RNN)-np.min(h_RNN))   
plt.plot(h_RNN,label='RNN')
############################# Model GRU ############################

# Path of the folder where the model paramters are saved
path = os.path.join(os.getcwd(),'best_model','GRU 4.1', 'best_params.pt')

# Initializing the model
model = GRU(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2, 
                dp_keep_prob=0.35).cuda()

#Setting the parameters to the one ontained in 4.1
model.load_state_dict(torch.load(path))

# Calculating the loss
run_epoch(model, valid_data, is_train=True, lr=1.0,grad_show=True)


h1_GRU=[]
for i in range(35):
    h1_GRU.append(torch.norm(torch.stack(model.gradients[2*i:2*i+1])))
    
h_GRU=torch.stack(h1_GRU) 
h_GRU=h_GRU.cpu().numpy()
h_GRU=(h_GRU-np.min(h_GRU))/(np.max(h_GRU)-np.min(h_GRU))   

plt.plot(h_GRU,label='GRU')

plt.plot(h_RNN,label='RNN')
plt.plot(h_GRU,label='GRU')
plt.legend()
plt.show()
