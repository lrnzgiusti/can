#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:18:07 2022

@author: ince
"""


#import sys
#sys.path.append("..")
#sys.path.append(".")
import graph_tool
import os
import pickle
import pathlib
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, QM9, ZINC, Planetoid


import torch_geometric.transforms as T
from .utils import compute_cell, ProgressParallel, compute_cell_complex_stat

from joblib import delayed


TU_DATASETS = ['MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MM', 'ENZYMES', 'COLORS-3']
CITATION_NETWORKS =  ['CORA', 'CITESEER', 'PUBMED']


class RenzIO(InMemoryDataset):
    def __init__(self, dataset_name, max_ring_size, compute_complex=False, split=None):
        super(RenzIO, self).__init__(dataset_name)
        self.max_ring_size = max_ring_size
        root = "./datasets"
        dataset_name = dataset_name.upper()
        if dataset_name in TU_DATASETS:
            dataset = TUDataset(root+"/TU/", dataset_name)
            connectivity_path = root + "/TU/" + dataset_name + "/cell_conn/"
        elif dataset_name == 'ZINC':
            dataset = ZINC(root+"/ZINC", subset=True, split=split)
            connectivity_path = root + "/ZINC/cell_conn/"+split+"/"
        elif dataset_name == 'qm9':
            dataset = QM9(root+"/QM9")
            connectivity_path = root + "/QM9/cell_conn/"
        elif dataset_name in CITATION_NETWORKS:
            dataset = Planetoid(root=root + "/CITATION/", name=dataset_name)
            connectivity_path = root + "/CITATION/" + dataset_name + "/cell_conn/"
            
        #make path to find connectivity information for the complex
        #print(dataset)
        #l = compute_cell_complex_stat(dataset, 6)
        #print({k:v/len(dataset) for k,v in dict(Counter(l)).items()})
        pathlib.Path(connectivity_path).mkdir(parents=False, exist_ok=True)    
        if compute_complex or "connectivity.pkl" not in os.listdir(connectivity_path):
            print("Starting Processing Dataset")
            
            parallel = ProgressParallel(n_jobs=1, use_tqdm=True, total=len(dataset)) 
            connectivity = parallel(delayed(compute_cell)(
                G, self.max_ring_size) for G in dataset)
            
            print("Processing Completed")
            
            pickle.dump(connectivity,  
                        open(connectivity_path+"connectivity.pkl", "wb"), 
                        protocol=pickle.HIGHEST_PROTOCOL)
        else:
            connectivity = pickle.load(open(connectivity_path+"connectivity.pkl", "rb"))
            
        self.data = dataset #remove .data for TU
        self.connectivities = connectivity
        
        
    def __getitem__(self, idx):
        return self.data[idx], self.connectivities[idx] 
        
    def __len__(self):
        return len(self.data)
    
