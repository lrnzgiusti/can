#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:00:00 2022

@author: ince
"""



from typing import Callable, TypeVar, Tuple, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
spmm = torch.sparse.mm

EdgeSignal = TypeVar('EdgeSignal')
Graph = TypeVar('Graph')

from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from utils.utils import readout




def sp_softmax(indices, values, N):
    """
    Compute the sparse softmax of the given values.

    Parameters
    ----------
    indices : torch.tensor
        The indices of the non-zero elements in the sparse tensor.
    values : torch.tensor
        The values of the non-zero elements in the sparse tensor.
    N : int
        The size of the output tensor.

    Returns
    -------
    softmax_v : torch.tensor
        The softmax values computed for the sparse tensor.
    """
    source, _ = indices
    v_max = values.max()
    exp_v = torch.exp(values - v_max)
    exp_sum = torch.zeros(N, 1, device='cuda')
    exp_sum.scatter_add_(0, source.unsqueeze(1), exp_v)
    exp_sum += 1e-10
    softmax_v = exp_v / exp_sum[source]
    return softmax_v


def sp_matmul(indices, values, mat):
    """
    Perform sparse matrix multiplication.
    Parameters
    ----------
    indices : torch.tensor
        The indices of the non-zero elements in the sparse tensor.
    values : torch.tensor
        The values of the non-zero elements in the sparse tensor.
    mat : torch.tensor
        The dense matrix to be multiplied with the sparse tensor.

    Returns
    -------
    out : torch.tensor
        The result of the sparse matrix multiplication.
    """ 
    source, target = indices
    out = torch.zeros_like(mat)
    out.scatter_add_(0, source.expand(mat.size(1), -1).t(), values * mat[target])
    return out


class CellLayer(nn.Module):
    """
    A Cell layer for a Cellular Attention Network (CAN).
    This layer is responsible for learning the cell update rules in the
    network using learnable weight matrices.

    Parameters
    ----------
    F_in : int
        Number of input features.
    F_out : int
        Number of output features.
    cell_forward_activation : Callable
        Non-linear activation function for the cell forward operation.
    cell_forward_dropout : float
        Dropout rate applied after the cell forward operation.
    param_init : str
        The initialization method for the learnable weight matrices.
        Choices: 'uniform', 'normal'

    Examples
    --------
    >>> cell_layer = CellLayer(F_in=10, F_out=20, cell_forward_activation=torch.relu, cell_forward_dropout=0.5, param_init='uniform')
    """

    def __init__(self, F_in: int, F_out: int,
                 cell_forward_activation: Callable,
                 cell_forward_dropout: float,
                 param_init: str):

        super(CellLayer, self).__init__()

        self.F_in = F_in
        self.F_out = F_out
        self.Wirr = nn.Parameter(torch.empty(size=(F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(F_in, F_out)))
        self.Wskip = nn.Parameter(torch.empty(size=(F_in, F_out)))
        self.W = nn.Parameter(torch.empty(size=(F_out, F_out)))
        self.Wout = nn.Parameter(torch.empty(size=(F_out, F_out)))

        self.param_init = param_init

        self.cell_forward_activation = cell_forward_activation
        self.cell_forward_dropout = cell_forward_dropout
        self.reset_parameters()

    def __repr__(self):
        s = "CellLayer(" + \
            "F_in=" + str(self.F_in) + \
            ", F_out=" + str(self.F_out) + \
            ", Fwd_Activ=" + str(self.cell_forward_activation) + \
            ", Fwd_Dropout=" + str(self.cell_forward_dropout) + ")"
        return s

    def reset_parameters(self):
        """
        Reinitialize the learnable weight matrices of the cell layer.
        The initialization method is determined by the `param_init` parameter.
        """
        gain = nn.init.calculate_gain('relu')

        if self.param_init == 'uniform':
            nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
            nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
            nn.init.xavier_uniform_(self.Wskip.data, gain=gain)
            nn.init.xavier_uniform_(self.W.data, gain=gain)
            nn.init.xavier_uniform_(self.Wout.data, gain=gain)
        else:
            nn.init.xavier_normal_(self.Wirr.data, gain=gain)
            nn.init.xavier_normal_(self.Wsol.data, gain=gain)
            nn.init.xavier_normal_(self.Wskip.data, gain=gain)
            nn.init.xavier_normal_(self.W.data, gain=gain)
            nn.init.xavier_normal_(self.Wout.data, gain=gain)



class TopologicalNorm(torch.nn.Module):
    def __init__(self, feat_dim, strategy):
        """
        
        Allow to perform data normalization by
        forward the also the complex information through the network
        
        the connectivity can also be ignored for the lift 
            
        Parameters
        ----------
        feat_dim : int
            dimension of the incomping signals' features.
        strategy : str
            Strategy to perform the normalization technique.

        Returns
        -------
        norm(x), G : EdgeSignal, Graph.

        """
        super(TopologicalNorm, self).__init__()
        assert (feat_dim > 0), "feature dimension of the signal must be  > 0"
        assert strategy in ['layer', 'batch', 'identity', 'id'], "TopologicalNorm strategy must be one of: ['layer', 'batch', 'identity']"
        if strategy == 'layer':
            self.tn=nn.LayerNorm(feat_dim)
        elif strategy == 'batch':
            self.tn=nn.BatchNorm1d(feat_dim)
        elif strategy == 'identity':
            self.tn=nn.Identity()

    def forward(self, x: Union[Tuple[EdgeSignal, Graph], EdgeSignal]):
        x, G = x
        return self.tn(x), G


class CellAttentionLayer(CellLayer):
    """
    Attention-based cell layer of Cellular Attention Network.
    
    This layer inherits from the CellLayer class and adds an attention mechanism
    for the information exchange between the edges weighted by learnable attention
    coefficients. 

    Parameters
    ----------
    F_in : int
        Number of input features for the cell attention layer.
    F_out : int
        Number of output features for the cell attention layer.
    skip : bool
        Whether to add skip connections in the attention layer.
    cell_attention_activation : Callable
        Non-linear activation function for the cell attention mechanism.
    cell_forward_activation : Callable
        Non-linear activation function for the forward pass of the cell layer.
    cell_attention_dropout : float
        Dropout rate applied to the attention mechanism.
    cell_forward_dropout : float
        Dropout rate applied to the forward pass of the cell layer.
    param_init : str
        Parameter initialization method, either 'uniform' or 'normal'.
    """

    def __init__(self, F_in: int, F_out: int, skip: bool,
                 cell_attention_activation: Callable,
                 cell_forward_activation: Callable,
                 cell_attention_dropout: float,
                 cell_forward_dropout: float,
                 param_init: str):

        # Call the constructor of the parent class (CellLayer)
        super(CellAttentionLayer, self).__init__(F_in=F_in,
                                                 F_out=F_out,
                                                 cell_forward_activation=cell_forward_activation,
                                                 cell_forward_dropout=cell_forward_dropout,
                                                 param_init=param_init)

        # Define learnable parameters for the attention mechanism
        self.att_irr = nn.Parameter(torch.empty(size=(2 * self.F_out, 1)))
        self.att_sol = nn.Parameter(torch.empty(size=(2 * self.F_out, 1)))

        self.param_init = param_init
        self.skip = skip

        self.cell_attention_activation = cell_attention_activation
        self.dropout = cell_attention_dropout

        # Reset and initialize parameters
        self.reset_parameters()
        
    def __repr__(self):
        cell_repr = super().__repr__()
        s = "AttentionLayer(" + \
            "Att Activ="+str(self.cell_attention_activation) +\
            ", Att Dropout="+str(self.dropout) +\
             ", Skip="+str(self.skip)   +")\n\t\t" + cell_repr + "\n)"
        return s
    
    def to(self, device):
        super().to(device)
        return self
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super().reset_parameters()
        gain = nn.init.calculate_gain('relu')
        if self.param_init == 'uniform':
            nn.init.xavier_uniform_(self.att_irr.data, gain=gain)
            nn.init.xavier_uniform_(self.att_sol.data, gain=gain)
        else:
            nn.init.xavier_normal_(self.att_irr.data, gain=gain)
            nn.init.xavier_normal_(self.att_sol.data, gain=gain)
    
    
    def forward(self, x: Tuple[EdgeSignal, Graph]) -> EdgeSignal:
       
        
        x, G = x
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        out = (1.001)*(x @ self.Wskip) if self.skip else torch.tensor(0.0)
        
        
        try:
            h = torch.matmul(x, self.Wirr)
            source, target = G.connectivities['do']
            a = torch.cat([h[source], h[target]], dim=1)
            e = self.cell_attention_activation(torch.matmul(a, self.att_irr))
            #e = F.dropout(e, self.dropout, training=self.training)
    
            attention = sp_softmax(G.connectivities['do'], e, h.size(0))
            attention = F.dropout(attention, self.dropout, training=self.training)
            #h = F.dropout(h, self.dropout, training=self.training)
            h_prime = sp_matmul(G.connectivities['do'], attention, h)
        except:
            h_prime = torch.tensor(0.0)
            
        out = h_prime 
        
        try: 
            h = torch.matmul(x, self.Wsol)
            source, target = G.connectivities['up']
            a = torch.cat([h[source], h[target]], dim=1)
            e = self.cell_attention_activation(torch.matmul(a, self.att_sol))
            #e = F.dropout(e, self.dropout, training=self.training)
    
            attention = sp_softmax(G.connectivities['up'], e, h.size(0))
            attention = F.dropout(attention, self.dropout, training=self.training)
            #h = F.dropout(h, self.dropout, training=self.training)
            h_prime = sp_matmul(G.connectivities['up'], attention, h)
        except:
            h_prime = torch.tensor(0.0)
        
        out += h_prime
    
        
        return self.cell_forward_activation(out), G



class MultiHeadCellAttentionLayer(nn.Module):

    """
    mhcal = MultiHeadCellAttentionLayer(F_in=3, F_out=3, 
                                        sigma=torch.nn.ELU(), K=3, 
                                        p_dropout=0.2)
    """
    def __init__(self, F_in: int, F_out: int, skip: bool,
                       cell_attention_heads: int, 
                       cell_attention_activation: Callable,
                       cell_forward_activation: Callable,
                       cell_attention_dropout: float,
                       cell_forward_dropout: float,
                       cell_attention_readout: str,
                       param_init: str):
        """
        cell_attention_activation: Callable, cell_attention_dropout: float,
        cell_attention_readout: str,
        cell_forward_activation: Callable, cell_forward_dropout: float,
        alpha_leaky_relu : float
            nevative slope of the leakyrelu.
            
        readout : str
            which function to use in order to perform readout operations
            
        K: Numer of attention heads
        """
        super(MultiHeadCellAttentionLayer, self).__init__()
        assert F_in > 0, ValueError("Number of input feature must be > 0")
        assert F_out > 0, ValueError("Number of output feature must be > 0")
        assert cell_attention_readout in ['cat', 'avg', 'sum', 'max'], ValueError("readout must be one of ['cat', 'avg', 'sum', 'max']")
        assert param_init in ['uniform', 'normal'], ValueError("Param init must be one of ['uniform', 'normal']")
        self.F_out = F_out
        self.F_in = F_in
        self.skip=skip
        self.cell_attention_heads=cell_attention_heads
        self.cell_attention_readout = cell_attention_readout
        self.cell_forward_activation = cell_forward_activation
        self.cell_forward_dropout = cell_forward_dropout
        self.attentions = [CellAttentionLayer(F_in=F_in,
                                              F_out=F_out,
                                              cell_attention_dropout=cell_attention_dropout,
                                              cell_attention_activation=cell_attention_activation,
                                              cell_forward_activation=cell_forward_activation,
                                              cell_forward_dropout=cell_forward_dropout,
                                              param_init=param_init,
                                              skip=skip)
                           for _ in range(cell_attention_heads)]
        
        
        
    def __repr__(self):
        _F_out =  self.F_out*self.cell_attention_heads \
                 if self.cell_attention_readout == 'cat' else self.F_out
        s = "MultiHeadCellAttentionLayer(" + \
            "F_in="+str(self.F_in)+ \
            ", F_out="+str(_F_out)+\
            ", Heads=" +str(self.cell_attention_heads) + \
            ", Readout=" +self.cell_attention_readout+ \
            "):"
        for idx, attention_layer in enumerate(self.attentions):
            s+= "\n\t(" +str(idx)+"): "+ str(attention_layer)
        return s 
    
    def to(self, device):
        self.attentions = nn.ModuleList([attention.to(device) for attention in self.attentions])
        return self
        
    def forward(self, x: Tuple[EdgeSignal, Graph], *args, **kwargs):
        G = x[1]
        H = [attention_layer(x, *args, **kwargs) for attention_layer in self.attentions] 
        H = [hidden_signal[0] for hidden_signal in H]
        if self.cell_attention_readout == 'cat':
            H = torch.cat(H, dim=1) # Xe
        elif self.cell_attention_readout == 'sum':
            H = torch.stack(H, dim=2).sum(dim=2)
        elif self.cell_attention_readout == 'avg':
            H = torch.stack(H, dim=2).mean(dim=2)
        elif self.cell_attention_readout == 'max':
            H = torch.stack(H, dim=2).max(dim=2).values
            
        return H, G


class CAPooLayer(nn.Module):
    """
    CAPooLayer (Cellular Attention Pooling Layer) is responsible for pooling
    operations in the Cellular Graph Attention Network.

    This layer applies attention-based pooling to a given edge signal
    and updates the graph accordingly.

    Parameters
    ----------
    k_pool : float
        Fraction of nodes to keep after the pooling operation.
    F_in : int
        Number of input features for the pooling layer.
    cell_forward_activation : Callable
        Non-linear activation function used in the forward pass.

    Returns
    -------
    CAPooLayer.

    Examples
    -------
    pool = CAPooLayer(k_pool=.75,
                      F_in=3*att_heads,
                      cell_forward_activation=nn.ReLU)
    """

    def __init__(self, k_pool: float, F_in: int, cell_forward_activation: Callable):
        super(CAPooLayer, self).__init__()

        self.k_pool = k_pool
        self.cell_forward_activation = cell_forward_activation

        # Learnable attention parameter for the pooling operation
        self.att_pool = nn.Parameter(torch.empty(size=(F_in, 1)))

        # Initialize the attention parameter using Xavier initialization
        nn.init.xavier_normal_(self.att_pool.data, gain=1.41)

        
    def __repr__(self):
       s = "PoolLayer(" + \
           "K Pool="+str(self.k_pool)+ ")"
       return s
    

    def forward(self,  x: EdgeSignal) -> EdgeSignal:
        
        x, G = x
        shape = x.shape
        Zp = x @ self.att_pool
        idx = topk(Zp.view(-1), self.k_pool, G.edge_batch)
        x = x[idx] * self.cell_forward_activation(Zp)[idx].view(-1, 1)
        G.edge_batch = G.edge_batch[idx]
        G.ros.append(readout(x, G.edge_batch, 'sum'))
        G.connectivities['up'] = tuple(filter_adj(torch.stack(G.connectivities['up']), None, idx, shape[0])[0])
        G.connectivities['do'] = tuple(filter_adj(torch.stack(G.connectivities['do']), None, idx, shape[0])[0])
        

        return x, G
