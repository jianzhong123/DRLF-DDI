import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import math
from torch_geometric.nn import GCNConv, RGCNConv, TransformerConv


class DDI_GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_layers):
        super().__init__()
        self.layer = num_layers
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads, dropout=0.1)
        self.conv2 = nn.ModuleList(
            [TransformerConv(hidden_channels * heads, hidden_channels, heads, dropout=0.1) for _ in
             range(2, num_layers)])

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index).relu()
        if self.layer > 1:
            for conv in self.conv2:
                x = conv(x, edge_index)  #
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output, attn = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X, attn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size, len_after_AE=128, bert_n_heads=4):
        super(AE1, self).__init__()

        self.vector_size = vector_size
        self.att2 = EncoderLayer(self.vector_size, bert_n_heads)

        self.l1 = torch.nn.Linear(self.vector_size, self.vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(self.vector_size // 2)

        # self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear(self.vector_size // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, self.vector_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.vector_size // 2)

        self.l4 = torch.nn.Linear(self.vector_size // 2, self.vector_size)

        self.dr = torch.nn.Dropout(0.1)
        self.ac = gelu  # nn.ReLU()#

    def forward(self, X):
        X, attn = self.att2(X)
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE


def min_max_normalize(feature1, feature2):
    min_val = feature1.min(dim=1, keepdim=True)[0]  # 按行最小值 (569, 1)
    max_val = feature1.max(dim=1, keepdim=True)[0]  # 按行最大值 (569, 1)

    feature2_norm = (feature2 - min_val) / (max_val - min_val)  # 避免除 0
    return feature2_norm


class model_S0(torch.nn.Module):
    def __init__(self, device, index, dim, l, head):  # in_dim,hidden_dim,out_dim,
        super(model_S0, self).__init__()
        self.device = device
        self.dim = dim
        self.l = l
        self.head = head

        self.trans2 = DDI_GraphTransformer(768, 128, self.head, self.l)

        self.mlp = nn.ModuleList([nn.Linear(512, 512),  # 2560
                                  nn.ELU(),
                                  nn.Linear(512, 65)
                                  ])

        morgen_path = '../data/drugfeatures/morgen_fingerprint_DDIMDL.npy'
        np_morgen = np.load(morgen_path)
        self.morgen = torch.tensor(np_morgen).to(device).float()
        # '../data/drugfeatures/S2_trim_DDIMDL_cross' + str(index) + '.npy'
        alignn_path = '../data/drugfeatures/molformer_DDIMDL.npy'  #
        alignn = np.load(alignn_path)
        self.alignn = torch.tensor(alignn).to(device).float()

        self.ae1 = AE1(1024, self.dim * head, self.head)  #

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def fuse_features(self, modal_feat, graph_feat):
        combined = torch.cat([modal_feat, graph_feat], dim=1)  # [batch, 2*feat_dim]
        gate = torch.sigmoid(self.fc_gate(combined))  # [batch, feat_dim]
        fused = gate * modal_feat + (1 - gate) * graph_feat  # [batch, feat_dim]
        return fused

    def forward(self, drug1s, drug2s,train_graph):
        modalfeatures1, XE1 = self.ae1(self.morgen)
        drug1_emb_morgen = modalfeatures1[drug1s.long()]
        drug2_emb_morgen =  modalfeatures1[drug2s.long()]

        # drug1_emb_morgen = self.morgen[drug1s.long()]
        # drug2_emb_morgen = self.morgen[drug2s.long()]
        #drug1_emb_mol = self.alignn[drug1s.long()]
        #drug2_emb_mol = self.alignn[drug2s.long()]

        graph_emb = self.trans2(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
        drug1_emb_graph = graph_emb[drug1s.long()]
        drug2_emb_graph = graph_emb[drug2s.long()]

        #all = torch.cat((drug1_emb_morgen, drug1_emb_graph, drug2_emb_morgen, drug2_emb_graph), 1)  #
        # all = self.bn(all)
        all = torch.cat((drug1_emb_graph, drug2_emb_graph), 1)
        #print(all.shape)
        out = self.MLP(all, 3)

        return out
