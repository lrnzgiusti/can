#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:43:56 2022

@author: ince
"""

import graph_tool

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import json
import argparse
import logging
import torch_geometric

from sklearn.model_selection import StratifiedKFold

from architecture.cell_network import CellNetwork
from utils.RenzIO import RenzIO
from utils.utils import collate_complexes
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



logging.getLogger("lightning").setLevel(logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

parser = argparse.ArgumentParser()


parser.add_argument("-c", "--config_file", help="config file to assign the arch params .json",
                    type=str, default="configs/config.json")
parser.add_argument("-f", "--fold", help="fold",
                    type=int, default=1)
parser.add_argument("-d", "--dataset", help="dataset name",
                    type=str, default='mutag')
parser.add_argument("-id", "--pci_id", help="id bus",
                    type=str, default="2")
parser.add_argument("-r", "--random", help="random search",
                    type=bool, default=False)

args = parser.parse_args()




device = torch.device(
    "cuda:"+args.pci_id if torch.cuda.is_available() else torch.device("cpu"))



config = json.load(open(args.config_file, "r"))
    
    


seed = 12091996

pl.seed_everything(12091996)
ns = config['negative_slope']

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(ns),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
    'leaky_relu' : nn.LeakyReLU(ns),
}

config['signal_lift_activation'] = activations[config['signal_lift_activation']]
config['cell_attention_activation'] = activations[config['cell_attention_activation']]
config['cell_forward_activation'] = activations[config['cell_forward_activation']]
max_ring_size = config['max_ring_size']
bs = config['bs']


#config['dataset'] = args.dataset #remove this, eventually
dataset_name = config['dataset']
#%%%


dataset = RenzIO(dataset_name, max_ring_size)



skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
labels = [G.y.item() for G in dataset.data]
idx_list = []
for idx in skf.split(np.zeros(len(labels)), labels):
    idx_list.append(idx)
train_idxs, test_idxs = idx_list[args.fold]



train_sampler = SubsetRandomSampler(train_idxs)
test_sampler = SubsetRandomSampler(test_idxs)


train_dataloader = DataLoader(dataset, sampler=train_sampler, 
                              collate_fn=collate_complexes, batch_size=bs,
                              num_workers=0, pin_memory=True)
test_dataloader = DataLoader(dataset, sampler=test_sampler, 
                             collate_fn=collate_complexes, batch_size=bs,
                             num_workers=0, pin_memory=True)


#%%


s = CellNetwork(in_features={'node':dataset.data.num_features,
                             'edge':dataset.data.num_edge_features}, #
                n_class=dataset.data.num_classes,
                **config, device=device).to(device)


early_stop_callback = EarlyStopping(monitor="valid_acc", min_delta=0.001, patience=200,
                                    verbose=True, mode="max")
    
    
logger = pl.loggers.TensorBoardLogger(name=config['dataset'], save_dir='results')


trainer = pl.Trainer(max_epochs=config['max_epochs'], logger=logger,callbacks=[early_stop_callback],
                     devices=[int(args.pci_id)], accelerator="gpu", auto_select_gpus=False)



trainer.fit(s, train_dataloader, test_dataloader)


config['val_acc'] = max(s.valid_acc_epoch)
config['train_acc'] = max(s.train_acc_epoch)
config['fold'] = int(args.fold)
config['signal_lift_activation'] = str(config['signal_lift_activation'])
config['cell_attention_activation'] =  str(config['cell_attention_activation'])
config['cell_forward_activation'] =  str(config['cell_forward_activation'])

with open("results/cfg2acc_"+dataset_name+".txt", "a") as fout:
    fout.write(json.dumps(config))
    fout.write("\n")
