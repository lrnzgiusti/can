#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
    """
    A Cellular version of Graph Attention Networks that lifts the node feature vectors
    of GNNs. The information is exchanged between the edges weighted by learnable attention
    coefficients.

    Attributes:
        lr (float): Learning rate
        wd (float): Weight decay
        readout (str): Readout strategy for the dense layer
        lift (nn.Sequential): MultiHeadLiftLayer and TopologicalNorm layers for signal lifting
        dense (List[int]): A list of dense layers
        N_dense_layers (int): The number of dense layers
        cell_net (nn.Sequential): Sequential Cellular Attention layers
        mlp (nn.Sequential): Multi-layer perceptron
        max_acc (float): Maximum accuracy
        loss_fn (nn.BCEWithLogitsLoss): Binary Cross-Entropy with Logits Loss function
        train_acc (List[float]): Training accuracy
        valid_acc (List[float]): Validation accuracy
        valid_acc_epoch (List[float]): Validation accuracy per epoch
        train_acc_epoch (List[float]): Training accuracy per epoch
    """
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
        """
        Initializes the CellNetwork.

        Args:
            in_features (int): The input features
            n_class (int): The number of classes
            features (List[int]): List of features for each layer
            cell_attention_heads (List[int]): List of cell attention heads
            dense (List[int]): List of dense layers
            norm_strategy (str): Normalization strategy
            signal_heads (int): Number of signal heads
            signal_lift_activation (Callable): Activation function for signal lifting
            signal_lift_dropout (float): Dropout rate for signal lifting
            signal_lift_readout (str): Readout strategy for signal lifting
            cell_attention_activation (Callable): Activation function for cell attention
            cell_attention_dropout (float): Dropout rate for cell attention
            cell_attention_readout (str): Readout strategy for cell attention
            cell_forward_activation (Callable): Activation function for forward pass
            cell_forward_dropout (float): Dropout rate for forward pass
            dense_readout (str): Readout strategy for dense layer
            skip (bool): Whether to use skip connections
            lr (float): Learning rate
            wd (float): Weight decay
            k_pool (float): k value for the pooling layer
            device (str): Device to run the model on
            param_init (str): Parameter initialization strategy
            **kwargs: Additional keyword arguments
        """
        super(CellNetwork, self).__init__()

        self.lr = lr
        self.wd = wd
        self.readout = dense_readout

        # Initialize lift layers
        lift = MultiHeadLiftLayer(F_in=in_features['node'], 
                                  K=signal_heads, 
                                  signal_lift_activation=signal_lift_activation,
                                  signal_lift_dropout=signal_lift_dropout,
                                  signal_lift_readout=signal_lift_readout).to(device)
        lift_out_feat = signal_heads    
        norm = TopologicalNorm(feat_dim=lift_out_feat, 
                               strategy=norm_strategy)
        self.lift = nn.Sequential(lift, norm)

        # Initialize cell attention layers
        in_features = [lift_out_feat+ in_features['edge']] + \
                      [features[i]*cell_attention_heads[i]
                       if cell_attention_readout == 'cat'
                       else features[i]
                       for i in range(len(features))]
        self.dense = dense
        self.dense.append(n_class)
        self.N_dense_layers = len(dense)
        ops = []

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

            ops.append(layer)
            ops.append(can_norm)

            if l == len(in_features)-2:
                pool = CAPooLayer(k_pool=k_pool,
                                  F_in=CAN_out_feat, 
                                  cell_forward_activation=cell_forward_activation)
                ops.append(pool)

        # Initialize MLP
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
        self.train_acc = []
        self.valid_acc = []
        self.valid_acc_epoch = []
        self.train_acc_epoch = []
        print(self)

    def to(self, device):
        """
        Moves the CellNetwork to the specified device.

        Args:
            device (str): The device to move the CellNetwork to

        Returns:
            self: The CellNetwork after being moved to the device
        """
        super().to(device)
        self.cell_net = self.cell_net.to(device)
        self.lift = self.lift.to(device)
        self.mlp = self.mlp.to(device)
        return self

    def forward(self, G: Graph):
        """
        Forward pass of the CellNetwork.

        Args:
            G (Graph): Input graph

        Returns:
            h_mlp (torch.Tensor): Output tensor after the forward pass
        """
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
        """
        Propagate the batch through the model.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            loss (torch.Tensor): Loss for the batch
            acc (torch.Tensor): Accuracy for the batch
        """
        G = batch
        row, col = G.edge_index
        G.edge_batch = G.batch[row]
        y_hat =  self(G)
        loss = self.loss_fn(y_hat, torch.nn.functional.one_hot(G.y, num_classes=2).float())    

        acc = ((y_hat.argmax(dim=1) == G.y)*1).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            loss (torch.Tensor): Loss for the training batch
        """
        loss, acc = self.propagate(batch, batch_idx)

        self.train_acc.append(acc)

        self.log('train_loss', loss.item(), on_step=False,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            loss (torch.Tensor): Loss for the validation batch
        """
        loss, acc = self.propagate(batch, batch_idx)
        self.valid_acc.append(acc)
        self.log('valid_loss', loss.item(), on_step=False,
                 on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            loss (torch.Tensor): Loss for the test batch
        """
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        """
        Operations at the end of a training epoch.

        Args:
            outs (List): Output list
        """
        epoch_train_acc = torch.tensor(self.train_acc, dtype=torch.float).mean().item()
        self.train_acc_epoch.append(epoch_train_acc)
        self.log('train_acc', epoch_train_acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.train_acc = []

    def validation_epoch_end(self, outs):
        """
        Operations at the end of a validation epoch.

        Args:
            outs (List): Output list
        """
        epoch_valid_acc = torch.tensor(self.valid_acc, dtype=torch.float).mean().item()
        self.valid_acc_epoch.append(epoch_valid_acc)
        self.log('valid_acc', epoch_valid_acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.valid_acc = []

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer, learning rate scheduler, and monitor
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',
                                      factor=0.5,
                                      patience=50,
                                      min_lr=7e-5,
                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_acc'}

