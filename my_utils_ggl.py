import os.path as osp
#无CitationFull,WikiCS,AttributedGraphDataset,StochasticBlockModelDataset
from gammagl.datasets import Planetoid, Coauthor, Amazon
import gammagl.transforms as T
import torch
# from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import random_split
import matplotlib.pyplot as plt
import pickle
from time import perf_counter as t

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Computers', 'Photo', 'ogbn-arxiv', 'ogbg-code', 'BlogCatalog', 'Flickr', 'sbm']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    # if name == 'WikiCS':
    #     return WikiCS(root=osp.join(root_path, name))

    if name == 'Computers':
        return Amazon(root=root_path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Photo':
        return Amazon(root=root_path, name='photo', transform=T.NormalizeFeatures())
    # if name in ['BlogCatalog']:
    #     return AttributedGraphDataset(root=root_path, name=name, transform=T.NormalizeFeatures())
    # if name in ['Flickr']:
    #     return AttributedGraphDataset(root=root_path, name=name, transform=NormalizeFeaturesSparse())
    # if name.startswith('ogbn'):
    #     return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures()) # public split

from typing import List, Union

# from gammagl import Data,HeteroData
# from torch_geometric.data.datapipes import functional_transform
# from gammagl.transforms import BaseTransform

# # @functional_transform('normalize_features_sparse')
# class NormalizeFeaturesSparse(BaseTransform):
#     r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
#     (functional name: :obj:`normalize_features`).

#     Args:
#         attrs (List[str]): The names of attributes to normalize.
#             (default: :obj:`["x"]`)
#     """
#     def __init__(self, attrs: List[str] = ["x"]):
#         self.attrs = attrs

#     def __call__(
#         self,
#         data: Union[Data, HeteroData],
#     ) -> Union[Data, HeteroData]:
#         for store in data.stores:
#             for key, value in store.items(*self.attrs):
#                 value = value.to_dense()
#                 value = value - value.min()
#                 value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
#                 store[key] = value
#         return data

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(tlx.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = tlx.zeros((num_samples,)).to(tlx.bool)
    test_mask = tlx.zeros((num_samples,)).to(tlx.bool)
    val_mask = tlx.zeros((num_samples,)).to(tlx.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_alpha_beta(l, u, alpha):
    alpha_L= tlx.zeros(l.shape,device=l.device)
    # alpha_U, beta_L, beta_U = torch.clone(alpha_L), torch.clone(alpha_L), torch.clone(alpha_L)
    alpha_U= tlx.zeros(l.shape,device=l.device)
    beta_L= tlx.zeros(l.shape,device=l.device)
    beta_U= tlx.zeros(l.shape,device=l.device)
    pos_mask = l >= 0
    neg_mask = u <= 0
    alpha_L[pos_mask] = 1
    alpha_U[pos_mask] = 1
    alpha_L[neg_mask] = alpha
    alpha_U[neg_mask] = alpha
    not_mask = ~(pos_mask | neg_mask)
    alpha_not_upp = u[not_mask] - alpha * l[not_mask]
    alpha_not = alpha_not_upp / (u[not_mask] - l[not_mask])
    alpha_L[not_mask] = alpha_not
    alpha_U[not_mask] = alpha_not
    beta_U[not_mask] = (alpha - 1) * u[not_mask] * l[not_mask] / alpha_not_upp
    return alpha_L, alpha_U, beta_L, beta_U

def get_crown_weights(l1, u1, l2, u2, alpha, gcn_weights, Wcl):
    alpha_2_L, alpha_2_U, beta_2_L, beta_2_U = get_alpha_beta(l2, u2, alpha) # onehop
    alpha_1_L, alpha_1_U, beta_1_L, beta_1_U = get_alpha_beta(l1, u1, alpha) # twohop
    lambda_2 = tlx.where(Wcl >= 0, alpha_2_L, alpha_2_U) # N * d
    Delta_2 = tlx.where(Wcl >= 0, beta_2_L, beta_2_U) # N * d
    Lambda_2 = lambda_2 * Wcl # N * d
    W1_tensor, b1_tensor, W2_tensor, b2_tensor = gcn_weights
    W_tilde_2 = Lambda_2 @ W2_tensor.T
    b_tilde_2 = tlx.diag(Lambda_2 @ (Delta_2 + b2_tensor).T)
    lambda_1 = tlx.where(W_tilde_2 >= 0, alpha_1_L, alpha_1_U)
    Delta_1 = tlx.where(W_tilde_2 >= 0, beta_1_L, beta_1_U)
    Lambda_1 = lambda_1 * W_tilde_2
    W_tilde_1 = Lambda_1 @ W1_tensor.T
    b_tilde_1 = tlx.diag(Lambda_1 @ (Delta_1 + b1_tensor).T)
    return W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2
def get_batch(node_list, batch_size, epoch):
    num_nodes = len(node_list)
    num_batches = (num_nodes - 1) // batch_size + 1
    i = epoch % num_batches
    if (i + 1) * batch_size >= len(node_list):
        node_list_batch = node_list[i * batch_size:]
    else:
        node_list_batch = node_list[i * batch_size:(i + 1) * batch_size]
    return node_list_batch
def get_A_bounds(dataset, drop_rate):
    upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{dataset}_{drop_rate}_upper_lower.pkl")
    if osp.exists(upper_lower_file):
        pickle.load(upper_lower_file)
        # A_upper, A_lower = tlx.model.load_weights(upper_lower_file)
    else:
        A_upper, A_lower = None, None
    return A_upper, A_lower
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

from typing import Optional, Tuple

def filter_adj(row, col, edge_attr,
               mask):
    mask = tlx.convert_to_tensor(mask, dtype=tlx.bool)
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

def dropout_adj(
    edge_index:tlx.convert_to_tensor,
    edge_attr:tlx.convert_to_tensor= None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    # row, col = edge_index
    row = edge_index[0]
    col = edge_index[1]

    mask = torch.rand(row.numpy().size) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = tlx.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = tlx.stack([row, col])

    return edge_index, edge_attr


def to_dense_adj(
    edge_index,
    batch:tlx.convert_to_tensor = None,
    edge_attr:tlx.convert_to_tensor = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
):

    if batch is None:
        num_nodes = int(tlx.reduce_max(edge_index)) + 1 if tlx.numel(edge_index) > 0 else 0
        # batch = edge_index.new_zeros(num_nodes)
        batch = tlx.zeros(num_nodes)

    if batch_size is None:
        batch_size = int(tlx.reduce_max(batch)) + 1 if tlx.numel(batch) > 0 else 1

    one = tlx.ones(batch.shape[0])
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = tlx.cumsum(num_nodes)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(tlx.reduce_max(num_nodes))

    elif ((tlx.numel(idx1) > 0 and tlx.reduce_max(idx1) >= max_num_nodes)
          or (tlx.numel(idx2) > 0 and tlx.reduce_max(idx2) >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = tlx.ones(tlx.numel(idx0))

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = adj.reshape(size)

    return adj