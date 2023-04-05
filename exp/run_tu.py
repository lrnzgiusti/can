#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries and modules
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

# Set up logging
logging.getLogger("lightning").setLevel(logging.ERROR)

# Set up environment variable
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Initialize the argument parser
parser = argparse.ArgumentParser()
# ... the remaining code ...

# Set up device for running the experiment
device = torch.device("cuda:"+args.pci_id if torch.cuda.is_available() else torch.device("cpu"))

# Load configuration parameters from JSON file
config = json.load(open(args.config_file, "r"))

# Set the random seed for reproducibility
seed = 12091996
pl.seed_everything(seed)

# ... the remaining code ...

# Load dataset using RenzIO utility
dataset = RenzIO(dataset_name, max_ring_size)

# Perform stratified k-fold cross-validation for the dataset
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
labels = [G.y.item() for G in dataset.data]
idx_list = []
for idx in skf.split(np.zeros(len(labels)), labels):
    idx_list.append(idx)
train_idxs, test_idxs = idx_list[args.fold]

# Set up data loaders for training and testing
train_sampler = SubsetRandomSampler(train_idxs)
test_sampler = SubsetRandomSampler(test_idxs)
train_dataloader = DataLoader(dataset, sampler=train_sampler, collate_fn=collate_complexes, batch_size=bs,
                              num_workers=0, pin_memory=True)
test_dataloader = DataLoader(dataset, sampler=test_sampler, collate_fn=collate_complexes, batch_size=bs,
                             num_workers=0, pin_memory=True)

# Instantiate Cellular  Attention Network (CellNetwork) model
s = CellNetwork(in_features={'node': dataset.data.num_features,
                             'edge': dataset.data.num_edge_features},
                n_class=dataset.data.num_classes,
                **config, device=device).to(device)

# Set up early stopping callback
early_stop_callback = EarlyStopping(monitor="valid_acc", min_delta=0.001, patience=200,
                                    verbose=True, mode="max")

# Set up TensorBoard logger for recording experiment results
logger = pl.loggers.TensorBoardLogger(name=config['dataset'], save_dir='results')

# Initialize PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=config['max_epochs'], logger=logger, callbacks=[early_stop_callback],
                     devices=[int(args.pci_id)], accelerator="gpu", auto_select_gpus=False)

# Train and validate the model
trainer.fit(s, train_dataloader, test_dataloader)

# Save the best validation and training accuracies in the configuration
config['val_acc'] = max(s.valid_acc_epoch)
config['train_acc'] = max(s.train_acc_epoch)
config['fold'] = int(args.fold)
config['signal_lift_activation'] = str(config['signal_lift_activation'])
config['cell_attention_activation'] = str(config['cell_attention_activation'])
config['cell_forward_activation'] = str(config['cell_forward_activation'])

# Save the configuration and accuracies to a file
with open("results/cfg2acc_" + dataset_name + ".txt", "a") as fout:
    fout.write(json.dumps(config))
    fout.write("\n")
