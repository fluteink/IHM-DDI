from argparse import Namespace
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.nn.parameter import Parameter


class MSDGCL(nn.Module):
    def __init__(self, args):
        super(MSDGCL, self).__init__()
        self.feat_encoder = FeatureEncoder(args)
        self.graph_encoder = GraphEncoder(args)
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()
        self.tau = args.tau
        self.proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            self.activation,
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.sigm = nn.Sigmoid()
        self.multi_head_att_fusion = GlobalFeatureAttention(hidden_dim=args.hidden_dim * 3, num_heads=4)
        self.mse_loss_fn = nn.MSELoss()
        self.dnn_dropout_rate = args.dnn_dropout
        self.predictor = nn.Sequential(
            nn.Linear(args.hidden_dim * 3, args.drug_nums),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.drug_nums),
            self.activation,
            nn.Linear(args.drug_nums, args.drug_nums)
        )
        self.sigmoid = nn.Sigmoid()

    def build_knn_graph(self, features, k=10):
        features = features.cpu().detach().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        distances, indices = nbrs.kneighbors(features)
        num_nodes = features.shape[0]
        adj_matrix = sp.lil_matrix((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(1, k + 1):
                adj_matrix[i, indices[i, j]] = 1
                adj_matrix[indices[i, j], i] = 1
        adj_knn_norm = normalize_adj(adj_matrix.tocsr())
        adj_knn_norm = sparse_mx_to_torch_sparse_tensor(adj_knn_norm)
        return adj_knn_norm.cuda()

    def forward(self, x, adj, adj_diff):
        z1, z2, z3, x_d = self.feat_encoder(x)
        f_d = torch.cat((z1, z2, z3), dim=-1)
        adj_knn = self.build_knn_graph(f_d)
        f_t, f_knn, f_diff = self.graph_encoder(f_d, adj, adj_knn, adj_diff)
        mse_loss = self.mse_loss_fn(x_d, x)
        emb = torch.cat((f_t, f_knn, f_diff), dim=-1)
        emb = self.multi_head_att_fusion(emb)
        pred = self.sigmoid(self.predictor(emb))
        return pred, mse_loss, emb


class FeatureEncoder(nn.Module):
    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()
        self.hidden1 = nn.Sequential(
            nn.Linear(args.feature_dim, args.hidden_dim * 4),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim * 4),
            self.activation,
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(args.hidden_dim * 4, args.hidden_dim * 2),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim * 2),
            self.activation,
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim),
            self.activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim * 2),
            self.activation,

            nn.Linear(args.hidden_dim * 2, args.hidden_dim * 4),
            self.activation,

            nn.Linear(args.hidden_dim * 4, args.feature_dim),  # 输出与输入维度一致
            self.activation,
        )

    def forward(self, x):
        z1 = self.hidden1(x)
        z2 = self.hidden2(z1)
        z3 = self.hidden3(z2)
        x_ = self.decoder(z3)
        return z1, z2, z3, x_


class GraphEncoder(nn.Module):
    def __init__(self, args: Namespace):
        super(GraphEncoder, self).__init__()
        self.dropout = nn.Dropout(args.dnn_dropout)
        self.gnn1 = GraphConvolution(in_features=args.hidden_dim * 7, out_features=args.hidden_dim * 4)
        self.gnn2 = GraphConvolution(in_features=args.hidden_dim * 4, out_features=args.hidden_dim * 2)
        self.gnn3 = GraphConvolution(in_features=args.hidden_dim * 2, out_features=args.hidden_dim)
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()

    def forward(self, x, adj, adj_knn, adj_diff):
        # GNN-1
        h1 = self.activation(self.gnn1(x, adj))
        hk1 = self.activation(self.gnn1(x, adj_knn))
        hd1 = self.activation(self.gnn1(x, adj_diff))

        # GNN-2
        h2 = self.activation(self.gnn2(h1, adj))
        hk2 = self.activation(self.gnn2(hk1, adj_knn))
        hd2 = self.activation(self.gnn2(hd1, adj_diff))

        # GNN-2
        h3 = self.activation(self.gnn3(h2, adj))
        hk3 = self.activation(self.gnn3(hk2, adj_knn))
        hd3 = self.activation(self.gnn3(hd2, adj_diff))

        # return torch.cat((h1, h2, h3), dim=-1), torch.cat((hk1, hk2, hk3), dim=-1), torch.cat((hd1, hd2, hd3), dim=-1)
        return h3, hk3, hd3


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, inputs: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):  # ()
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  #
        h_prime = torch.matmul(attention, Wh)  #
        # return h_prime, attention
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (572,1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.t()
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GlobalFeatureAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q/K/V 投影（每个头独立）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        # 升维: (B, D) → (B, 1, D)
        x_seq = x.unsqueeze(1)

        # 生成 Q/K/V (每个头独立)
        q = self.q_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)
        k = self.k_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)
        v = self.v_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)

        # 注意力计算 (单元素序列)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, 1, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, 1, C)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        attn_output = self.out_proj(attn_output.squeeze(1))  # (B, D)

        # 残差连接 + 层归一化
        return self.norm(x + attn_output)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


def sparse_to_tuple(sparse_mx: sp.dia_matrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_mx_to_torch_sparse_tensor(sparse_mx) \
        -> torch.sparse.torch.sparse_coo_tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.torch.sparse_coo_tensor(indices, values, shape)


def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    # eliminate self-loop
    adj_ = adj
    rowsum = np.array(adj_.sum(0))
    rowsum_power = []
    for i in rowsum:
        for j in i:
            if j != 0:
                j_power = np.power(j, -0.5)
                rowsum_power.append(j_power)
            else:
                j_power = 0
                rowsum_power.append(j_power)
    rowsum_power = np.array(rowsum_power)
    degree_mat_inv_sqrt = sp.diags(rowsum_power)

    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_norm
