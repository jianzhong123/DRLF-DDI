import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np

def construct_trans_graph(index):
    """ 读取 DDI 训练数据 """
    train_df = pd.read_csv(f'../data/Dengcross5datasets/Deng_S0/S0_DDIMDL_idx_train_cross_{index}.csv',header=0)

    train_edges = torch.tensor(train_df[['d1', 'd2']].values, dtype=torch.long).T  # (2, num_edges)

    # 关系类型（DDI interaction type）
    train_edge_attr = torch.tensor(train_df['type'].values, dtype=torch.long)

    # 读取药物特征
    drug_features = np.load(f'../data/drugfeatures/molformer_DDIMDL.npy')
    drug_features = torch.tensor(drug_features, dtype=torch.float)

    # 构造图数据
    train_graph = Data(x=drug_features, edge_index=train_edges, edge_attr=train_edge_attr)

    return train_graph