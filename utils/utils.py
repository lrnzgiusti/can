#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:06:24 2022

@author: ince
"""

##stick graph_tool on top to avoid segfaults

import graph_tool as gt
import graph_tool.topology as top

gt.openmp_set_num_threads(18)#os.cpu_count()-2)

import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

import gudhi as gd
import itertools
import networkx as nx

from typing import List, Dict
from torch import Tensor
from torch_scatter import scatter


from tqdm.auto import tqdm
from joblib import Parallel
from copy import deepcopy
from collections import defaultdict
from torch_geometric.data import Batch


def readout(value:torch.Tensor, labels:torch.LongTensor, op:str) -> (torch.Tensor, torch.LongTensor):
    """Group-wise average for (sparse) grouped tensors
    
    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)
    
    Returns: 
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)
        
    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2, 0.2, 0.2],    #-> group / class 3
                             [0.4, 0.4, 0.4],    #-> group / class 3
                             [0.0, 0.0, 0.0]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)
        
        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.3000, 0.3000, 0.3000]])
            
        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}
    
    labels = torch.LongTensor(list(map(key_val.get, labels)))
    
    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1)).cuda()
    
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, value)
    if op == 'avg':
        result = result / labels_count.float().unsqueeze(1)
    
    return result


class SparseDropout(torch.nn.Module):
    def __init__(self, p_droput=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-p_droput

    def forward(self, x, training):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse_coo_tensor(rc, val, x.shape)

def compute_projection_matrix(L, eps, kappa):
    P = (torch.eye(L.shape[0]) - eps*L)
    for _ in range(kappa):
        P = P @ P  # approximate the limit
    return P

def normalize(L, half_interval=False):
    assert(L.shape[0] == L.shape[1])
    topeig = torch.linalg.eigvalsh(L.to_dense()).max().item()
    values = L.values()
    if half_interval:
        values *= 1.0/topeig
    else:
        values *= 2.0/topeig

    return torch.sparse_coo_tensor(L.indices(), values,   size=L.shape).to_dense()


def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

def normalize2(L,Lx, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]    # we use the maximal eigenvalue of L to normalize
    #print("Topeig = %f" %(topeig))

    ret = Lx.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)
        
    return ret




def normalize3(L, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]   
    #print("Topeig = %f" %(topeig))

    ret = L.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return ret



def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)
    #
    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping.
    #(m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)



###### RING UTILS




def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)] # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim+2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim+2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex), so that is very efficient for us.
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
    """
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim+1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y

# ---- support for rings as cells Graph add_edge_list remove_parallel_edges

def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    
    gt.stats.remove_self_loops(graph_gt) 
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    cell_tables, id_maps = build_tables(simplex_tree, size)
    
    # Find rings in the graph
    rings = get_rings(edge_index, max_k=max_k)
    
    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                            simplex_tree, id_maps, simplex_tree.dimension())
    
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
                boundaries_tables[2][-1].append(id_maps[1][boundary])
    
    return boundaries_tables, boundaries, coboundaries


def compute_incidences(edges, max_k=7):
    """
    
    Get cellular incidence matrices from a graph.
    
    Parameters
    ----------
    g : dgl.heterograph.DGLHeteroGraph or edgelist
        Graph or Batched Graph.
    max_k : int, optional
        max ring size. The default is 7.

    Returns
    -------
    B1 & B2: torch.SparseTensor
        B1: node -> edge map
        B2: edge -> cell map


    
    #extract edge list
    try:
        edges = g.all_edges() 
        edges = torch.stack(edges, dim=0)
    except:
        edges=torch.tensor(g)
        s = edges[:,0]
        t = edges[:,1]
        edges = torch.stack((s,t), dim=0)
    """
    
    #from ([sources], [targets]) to ((s,t)) for s in sources and t in targets, dim=0)
    
    size = max(edges[0].max() , edges[1].max()) + 1 
    
    #compute simplex treem, utility for getting boundaries
    simplex_tree = pyg_to_simplex_tree(edges, size)
    
    #here we get the mapping from the cell_i to it's id (label)
    #dim(cell_i) in [0,1,2]
    _, id_maps =  build_tables_with_rings(edges, simplex_tree, size, max_k)
    
    #get the boundaries for computing B2
    _, boundaries, _ = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)
    
    ## build the inverted id map:
    ## inv_id_maps[i] = [cell_j] i in [0,1,2], j in [0, ..., len(inv_id_maps[i])-1]
    inv_id_maps = [[cell for cell,id_map in id_map.items()] for id_map in id_maps  ]
    
    nV= max(id_maps[0].values())+1 # number of vertices
    nE = max(id_maps[1].values())+1 # number of edges
    try:
        nF = max(id_maps[2].values())+1 # number of faces (cells)
    except:
        nF = 1
    B1 = np.zeros((nV, nE))
    B2 = np.zeros((nE, nF))
    
    
    # edge orientation is coherent with the ordering 
    # of the vertices in the edge description
    for idx, e in enumerate(inv_id_maps[1]):
        B1[e[0],idx] = +1 
        B1[e[1],idx] = -1
        
    # rings ordering is coherent with the orientation of its boundaries
    for idx, (face, bonds) in enumerate(boundaries[2].items()):
        #circular sliding windows of length 2 of the boundaries of the ring
        pairs = [(face[i], face[i+1]) for i in range(len(face)-1)] + [(face[-1], face[0])]
        for idx_bond, bond in enumerate(bonds):
            if bond in pairs:
                B2[id_maps[1][bond],idx] = +1
            elif bond[::-1] in pairs:
                B2[id_maps[1][bond],idx] = -1 
        
        
    # easy to store tough to handle algebraic stuff
    B1 = torch.from_numpy(B1).to_sparse().float()
    B2 = torch.from_numpy(B2).to_sparse().float()
    return B1, B2


def compute_cell_complex_stat(data, max_k):
    cell_dim_list = []
    for G in data:
        _, B2 = compute_incidences(G.edge_index.cpu(), max_k)
        
        cell_dim_list.extend(B2.to_dense().abs().sum(dim=0).to(int).tolist())
        
    return cell_dim_list

def compute_cell(G, max_k):
    
    B1, B2 = compute_incidences(G.edge_index.cpu(), max_k)
    
    
    #removes self loops from the node adjacency
    adj = torch.sparse.mm(B1, B1.t()).to_dense()
    adj = (adj - adj.diagonal()*torch.eye(adj.shape[0])).to_sparse() #remove self loops
    #this will be used as indexer in the signal lift attention mechanism
    edge_indices =  adj.indices() 
    
    #arranging the edge indices so that in lift phase 
    #is possible to index the connectivity directly and reshape the tensor
    #ysince triu indices are placed right before tril indices
    #permutation equivariance ensures symmetries
    """
    num_nodes = G.num_nodes
    triu = torch.triu_indices(num_nodes,num_nodes).T
    idxs_u = []
    idxs_l = []
    for idx in range(adj._nnz()):
        if (edge_indices[:, idx] == triu).all(axis=1).any(): #check if edge is in the triu matrix
            idxs_u.append(idx)
        else:
            idxs_l.append(idx)
                
    edge_indices = torch.hstack((edge_indices[:, idxs_u], 
                                 edge_indices[:, idxs_l]))
    """
    #remove self loops from the lower adjacency neighorhood
    Ldo = torch.sparse.mm(B1.t(), B1).coalesce().to_dense()#.fill_diagonal_(1).to_sparse()#.to_dense()
    Ldo = (Ldo - Ldo.diagonal()*torch.eye(Ldo.shape[0])).to_sparse() #remove self loops
    
    #remove self loops from the upper adjacency neighorhood
    Lup = torch.sparse.mm(B2 , B2.t()).coalesce().to_dense()#.fill_diagonal_(1).to_sparse()
    Lup = (Lup - Lup.diagonal()*torch.eye(Lup.shape[0])).to_sparse() #remove self loops
    
    
    ###
    ### Convolutions require the entire connectivity information of the complex
    ### Attention deals only with the connectivity structure of the complex
    ### To deal with graph batching we incorporate additional information to the graphs
    ### The connectivity information will be collated and adjusted according to the reindexing mechanism of the collator
    
    lower_neigh_connection = Ldo.coalesce().indices()
    upper_neigh_connection = Lup.coalesce().indices()
    
    return      {'do': lower_neigh_connection, 
                 'up': upper_neigh_connection, 
                 'adj': edge_indices}
                 #'P': compute_projection_matrix(normalize((Lup+Ldo).coalesce()), 0.88, 7)}

class ProgressParallel(Parallel):
    """A helper class for adding tqdm progressbar to the joblib library."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def collate_complexes(samples):
    """
    
    Each input graph becomes one disjoint component of the batched graph. The nodes
    and edges connecctivity matrices are relabeled to be disjoint segments:

    =================  =========  =================  ===  =========
                       graphs[0]  graphs[1]          ...  graphs[k]
    =================  =========  =================  ===  =========
    Original node ID   0 ~ N_0    0 ~ N_1            ...  0 ~ N_k
    New node ID        0 ~ N_0    N_0+1 ~ N_0+N_1+1  ...  1+\sum_{i=0}^{k-1} N_i ~
                                                          1+\sum_{i=0}^k N_i
    =================  =========  =================  ===  =========

    ------------------ EDGE REINDEXING ---------------------------------------------
    
    =================  =========  =================  ===  =========
                       graphs[0]  graphs[1]          ...  graphs[k]
    =================  =========  =================  ===  =========
    Original edge ID   0 ~ E_0    0 ~ E_1            ...  0 ~ E_k
    New edge ID        0 ~ E_0    E_0+1 ~ E_0+E_1+1  ...  1+\sum_{i=0}^{k-1} E_i ~
                                                          1+\sum_{i=0}^k E_i
    =================  =========  =================  ===  =========dataset.num_classes

    Parameters
    ----------
    samples : List of torch_geometric graphs with cell complex connectivity 

    Returns
    -------
    batched_graph:  InMemoryDataset
        batch complexes and adjust connectivity.
    """
    # collate for generating batched data
    # The samples is a list of pairs (graph, label)
    graphs, connectivities = map(list, zip(*samples))
    batched_connectivities = defaultdict(list)
    projection_matrices = []
    prev_num_nodes = prev_num_edges = 0
    for idx, graph in enumerate(graphs):
        batched_connectivities['adj'].append(connectivities[idx]['adj']+prev_num_nodes)
        batched_connectivities['do'].append(connectivities[idx]['do']+prev_num_edges)
        batched_connectivities['up'].append(connectivities[idx]['up']+prev_num_edges)
        prev_num_nodes += graph.num_nodes
        prev_num_edges += graph.num_edges
            

    batched_graph = Batch.from_data_list(graphs)
    batched_graph.connectivities = defaultdict(tuple)
    for k in batched_connectivities:
        batched_graph.connectivities[k] = tuple([torch.cat(x) for x in zip(*batched_connectivities[k])])
    
    batched_graph.ros = []
    batched_graph.default_connectivity = deepcopy(batched_graph.connectivities)
    
    #batched_graph.edge_attr = batched_graph.edge_attr.float().view(-1,1)
    #batched_graph.x = batched_graph.x.float()
    batched_graph.y = batched_graph.y.long()#.view(-1,1)
    return batched_graph
