#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:00:00 2022

@author: ince
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeVar, Tuple, Callable



NodeSignal = TypeVar('NodeSignal')
EdgeSignal = TypeVar('EdgeSignal')
Graph = TypeVar('Graph')


class LiftLayer(nn.Module):
    """
    A single lift layer for a Cell Attention Network (CAN).
    This layer is responsible for lifting node feature vectors into
    edge features using a learnable function

    Parameters
    ----------
    F_in : int
        Number of input features for the single lift layer.
    signal_lift_activation : Callable
        Non-linear activation function for the signal lift.
    signal_lift_dropout : float
        Dropout rate applied after the lift.

    Examples
    --------
    >>> lift_layer = LiftLayer(F_in=10, signal_lift_activation=torch.relu, signal_lift_dropout=0.5)
    """
    def __init__(self, F_in: int,
                 signal_lift_activation: Callable,
                 signal_lift_dropout: float):
        super(LiftLayer, self).__init__()

        self.F_in = F_in
        self.att = nn.Parameter(torch.empty(size=(2 * F_in, 1)))
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout
        self.reset_parameters()

    def __repr__(self):
        return "LiftLayer(" + \
               "F_in=" + str(self.F_in) + \
               ", Activation=" + str(self.signal_lift_activation) + \
               ", Dropout=" + str(self.signal_lift_dropout) + ")"

    def to(self, device):
        super().to(device)
        return self

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.att.data, gain=gain)

    def forward(self, x: Tuple[NodeSignal, Graph]) -> EdgeSignal:
        """
        Perform the forward pass for a single lift layer.

        Parameters
        ----------
        x : Tuple[NodeSignal, Graph]
            Input tuple containing the node signal (node feature vectors)
            and the graph structure (Graph object).

        Returns
        -------
        EdgeSignal
            The resulting edge signal after the lift layer operation.

        Notes
        -----
        The forward pass can be described as follows:
        1. Extract source and target nodes from the input graph's edge index.
        2. Concatenate source and target node feature vectors.
        3. Compute the output edge signal by applying the activation function to the
           matrix multiplication of the concatenated node features and the attention
           coefficients.
        """
        # Unpack input tuple into node signal and graph structure
        node_signal, graph = x

        # Extract source and target nodes from the graph's edge index
        source, target = graph.edge_index

        # Concatenate source and target node feature vectors
        node_features_stacked = torch.cat((node_signal[source], node_signal[target]), dim=1)

        # Compute the output edge signal by applying the activation function
        edge_signal = self.signal_lift_activation(node_features_stacked.mm(self.att))

        return edge_signal



class MultiHeadLiftLayer(nn.Module):

    def __init__(self, F_in:int, K:int, 
                 signal_lift_activation: Callable,
                 signal_lift_dropout: float,
                 signal_lift_readout: str, *args, **kwargs):
        """
        

        Parameters
        ----------
        F_in : int
            Number of input features for the lift layers.
        K : int
            Numer of attention heads.
        signal_lift_activation : Callable
            non-linear activation function for the signal lift.
        signal_lift_dropout : float
            dropout is applied after the lift.

        Returns
        -------
        Xe: torch.tensor
            A signal defined over the edges of the graph
        G: torch_geometric.dataset
            the complex tha

        """
        super(MultiHeadLiftLayer, self).__init__()
        
        self.F_in = F_in
        self.K = K
        self.signal_lift_readout = signal_lift_readout
        self.signal_lift_dropout = signal_lift_dropout
        self.signal_lift_activation = signal_lift_activation
        self.lifts = [LiftLayer(F_in=F_in, 
                                signal_lift_activation=signal_lift_activation, 
                                signal_lift_dropout=signal_lift_dropout) for _ in range(K)]
        
        """
        self.W = nn.Parameter(torch.empty(size=(K, 128)))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        """
        

    def __repr__(self):
        str_F_out = '2' if self.signal_lift_readout != 'cat' else  str(2*self.K)
        s = "MultiHeadLiftLayer(" + \
            "F_in="+str(self.F_in)+", F_out=" + str_F_out + \
            ", heads=" +str(self.K) + ", readout=" +self.signal_lift_readout+"):"
            
        for idx, lift in enumerate(self.lifts):
            s+= "\n\t(" +str(idx)+"): "+ str(lift)
        return s 
    
    def to(self, device):
        self.lifts = nn.ModuleList([lift.to(device) for lift in self.lifts])
        return self
        
    def forward(self, x):
        x, G = x
        Xe = [lift((x, G)) for lift in self.lifts] 
        if self.signal_lift_readout == 'cat':
            Xe = torch.cat(Xe, dim=1) # Xe
        elif self.signal_lift_readout == 'sum':
            Xe = torch.stack(Xe, dim=2).sum(dim=2)
        elif self.signal_lift_readout == 'avg':
            Xe = torch.stack(Xe, dim=2).mean(dim=2)
        elif self.signal_lift_readout == 'max':
            Xe = torch.stack(Xe, dim=2).max(dim=2).values
            
        
        Xe = F.dropout(Xe, self.signal_lift_dropout, training=self.training) #
        return Xe, G

