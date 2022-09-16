#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:32:51 2022

@author: ince
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:02:48 2022

@author: ince
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeVar, List, Callable
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.cell_layers import MultiHeadCellAttentionLayer, TopologicalNorm, CAPooLayer
from layers.signal_lift import MultiHeadLiftLayer
from utils.utils import compute_cell, readout


Graph = TypeVar('Graph')
Laplacian = TypeVar('Laplacian')
Signal = TypeVar('Signal')
ConnectivityMask = TypeVar('ConnectivityMask')
ReadoutIndexer = TypeVar('ReadoutIndexer')

class CellNetwork(pl.LightningModule):

    def __init__(self, in_features: int, n_class: int, 
                 features: List[int], cell_attention_heads: List[int], 
                 dense: List[int], norm_strategy: str, signal_heads: int, 
                 signal_lift_activation: Callable, signal_lift_dropout: float,
                 signal_lift_readout: str, 
                 cell_attention_activation: Callable, cell_attention_dropout: float,
                 cell_attention_readout: str,
                 cell_forward_activation: Callable, cell_forward_dropout: float,
                 dense_readout: str, skip: bool,
                 lr: float, wd: float,
                 k_pool: float, device: str, param_init: str, **kwargs):
        
        super(CellNetwork, self).__init__()

        

        self.lr = lr
        self.wd = wd
        self.readout = dense_readout
        ops = []
        """
        Lift is composed by:
            MultiHeadLiftLayer + Topological Norm
        """
        lift = MultiHeadLiftLayer(F_in=in_features['node'], 
                                  K=signal_heads, 
                                  signal_lift_activation=signal_lift_activation,
                                  signal_lift_dropout=signal_lift_dropout,
                                  signal_lift_readout=signal_lift_readout).to(device)
        
        lift_out_feat = signal_heads
                        
        
        norm = TopologicalNorm(feat_dim=lift_out_feat, 
                                        strategy=norm_strategy)
       
        
        self.lift = nn.Sequential(lift, norm)
        
        
        """
        
        Cell Attention is composed by:
            MultiHeadCellAttentionLayer + sigma + topological norm
        
        """
        # if you cat the heads the input features is f[i] * num_attention_heads[i]
        # if you readout the heads the number of input features to the layers is the default one
        in_features = [lift_out_feat+ in_features['edge']] + \
                        [features[i]*cell_attention_heads[i]
                        if cell_attention_readout == 'cat'
                        else features[i]
                        for i in range(len(features))
                        ] 
       
        
        
        self.dense = dense
        self.dense.append(n_class)

        self.N_dense_layers = len(dense)
        
        for l in range(len(in_features)-1):
            
            layer = MultiHeadCellAttentionLayer(cell_attention_heads=cell_attention_heads[l],
                                                F_in=in_features[l],
                                                F_out=features[l],
                                                skip=eval(skip),
                                                cell_attention_activation=cell_attention_activation,
                                                cell_attention_dropout=cell_attention_dropout,
                                                cell_forward_activation=cell_forward_activation,
                                                cell_forward_dropout=cell_forward_dropout,
                                                cell_attention_readout=cell_attention_readout,
                                                param_init=param_init).to(device)
         

            
            
            CAN_out_feat = features[l]*cell_attention_heads[l] \
                            if cell_attention_readout == 'cat' else features[l]
            
            
            can_norm = TopologicalNorm(feat_dim=CAN_out_feat, 
                                        strategy=norm_strategy)
            if False:
                pool = CAPooLayer(k_pool=k_pool,
                                      F_in=CAN_out_feat, 
                                      cell_forward_activation=cell_forward_activation)
                    
                
           
            ops.append(layer)
            ops.append(can_norm)
            #ops.append(pool)
            
            
            if l == len(in_features)-2:
                pool = CAPooLayer(k_pool=k_pool,
                                  F_in=CAN_out_feat, 
                                  cell_forward_activation=cell_forward_activation)
                
                ops.append(pool)
            
            
        
        
        mlp = []
        cell_to_dense_feat = features[-1]*cell_attention_heads[-1] \
                        if cell_attention_readout == 'cat' else features[-1]
        simplicial_to_dense = nn.Linear(
            cell_to_dense_feat, dense[0])
        mlp.extend([simplicial_to_dense])
        
        self.dropout = cell_forward_dropout

        for l in range(1, self.N_dense_layers):
            mlp.extend([cell_forward_activation, 
                        nn.BatchNorm1d(dense[l-1]),
                        nn.Dropout(cell_forward_dropout), 
                        
                        nn.Linear(dense[l-1], dense[l])])
                         
        
        
        self.cell_net = nn.Sequential(*ops)
        self.mlp = nn.Sequential(*mlp)
        
        
        self.max_acc = 0.0
        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = nn.L1Loss()
        self.train_acc = []
        self.valid_acc = []
        self.valid_acc_epoch = []
        self.train_acc_epoch = []
        print(self)

    def to(self, device):
        super().to(device)
        self.cell_net = self.cell_net.to(device)
        self.lift = self.lift.to(device)
        self.mlp = self.mlp.to(device)
        return self
        
    
    
    def forward(self, G: Graph):
        X = G.x
        Xe, _ = self.lift((X, G))
        
        if G.edge_attr is not None:
            Xe = torch.cat((Xe, G.edge_attr.float()), dim=1)
        
        H, _ = self.cell_net((Xe, G))
        
        H_ro = torch.stack(G.ros, dim=2).sum(dim=2)
        
        Xe = F.dropout(H_ro, self.dropout, training=self.training)
        h_mlp = self.mlp(H_ro)
        
        
        return h_mlp
    
    
    def propagate(self, batch, batch_idx):
    
        G = batch
        row, col = G.edge_index
        G.edge_batch = G.batch[row]
        y_hat =  self(G)
        loss = self.loss_fn(y_hat, torch.nn.functional.one_hot(G.y, num_classes=2).float())    
        
        acc = ((y_hat.argmax(dim=1) == G.y)*1).float().mean()
        return loss, acc
    
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.propagate(batch, batch_idx)

        self.train_acc.append(acc)
        
        self.log('train_loss', loss.item(), on_step=False,
                 on_epoch=True, prog_bar=True)
        
          
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        loss, acc = self.propagate(batch, batch_idx)
        self.valid_acc.append(acc)
        self.log('valid_loss', loss.item(), on_step=False,
                 on_epoch=True, prog_bar=True)
        
        
        return loss

        
    def test_step(self, batch, batch_idx):
       return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
       epoch_train_acc=torch.tensor(self.train_acc, dtype=torch.float).mean().item()
       self.train_acc_epoch.append(epoch_train_acc)
       self.log('train_acc',epoch_train_acc, on_step=False,
                on_epoch=True, prog_bar=True)
       self.train_acc = []

    def validation_epoch_end(self, outs):
        epoch_valid_acc=torch.tensor(self.valid_acc, dtype=torch.float).mean().item()
        self.valid_acc_epoch.append(epoch_valid_acc)
        #self.max_acc = max(self.valid_acc_epoch[1:])
        #print("\nMax Valid Acc:", self.max_acc, "\n")
        self.log('valid_acc',epoch_valid_acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.valid_acc = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',
                                      factor=0.5,
                                      patience=50,
                                      min_lr=7e-5,
                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_acc'}


