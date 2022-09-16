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
     Parameters
     ----------
     F_in : int
         Number of input features for the single lift layer.
     signal_lift_activation : Callable
         non-linear activation function for the signal lift.
     signal_lift_dropout : float
         dropout is applied after the lift.
    
    """
    def __init__(self, F_in: int, 
                       signal_lift_activation: Callable,
                       signal_lift_dropout: float):
       

        super(LiftLayer, self).__init__()

        self.F_in = F_in
        self.att =  nn.Parameter(torch.empty(size=(2*F_in, 1)))
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout  # 0.0#0.6
        self.reset_parameters()

    def __repr__(self):
        return "LiftLayer(" + \
            "F_in="+str(self.F_in)+\
            ", Activation=" +str(self.signal_lift_activation) + \
            ", Dropout=" +str(self.signal_lift_dropout)+")"
        
    def to(self, device):
        super().to(device)
        return self
        
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.att.data, gain=gain)

    def forward(self, x: Tuple[NodeSignal, Graph]) -> EdgeSignal:
        
        x, G = x
        
        
        s,t = G.edge_index
        x_st = torch.cat((x[s], x[t]), dim=1)
        out = self.signal_lift_activation(x_st.mm(self.att))
        
        """
        #tensor -- broadcast sum
        E = ((x @ self.att[:self.F_in]) + (
            x @ self.att[self.F_in:]).T) # (NxFin) + (FinxN) -> (NxN)
        
        x = E[G.connectivities['adj']].reshape(-1,1)
        
        return self.signal_lift_activation(F.dropout(x, self.signal_lift_dropout,
                                                     training=self.training))
        """
        return out




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

