from __future__ import division
from __future__ import print_function

import warnings

from scipy.sparse.linalg import inv, spsolve
from sklearn.neighbors import NearestNeighbors

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

torch.set_printoptions(threshold=1000000)
import torch.nn as nn
import numpy as np
import random

np.set_printoptions(threshold=10e6)
import pandas as pd
import scipy.sparse as sp
from argparse import ArgumentParser, Namespace
import os
import time
from typing import Union, Tuple

np.set_printoptions(threshold=np.inf)

from utils import get_roc_score
from utils import save_checkpoint, load_checkpoint
from utils import create_logger
from model import MSDGCL


# from features import get_features_generator, get_available_features_generators


def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='DeepDDI',
                        choices=['zhang', 'ChChMiner', 'DeepDDI'])
    parser.add_argument('--train_data_path', type=str, default='train.csv')
    parser.add_argument('--valid_data_path', type=str, default='valid.csv')
    parser.add_argument('--test_data_path', type=str, default='test.csv')

    # training
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--L2', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--activation', type=str, default='ELU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--dnn_dropout', type=float, default=0.1, help='Dropout rate of DNN.')

    parser.add_argument('--hidden_dim', type=int, default=32, help='Output dim')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1)

    # store
    parser.add_argument('--save_dir', type=str, default='./model_save',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')

    args = parser.parse_args()
    # config
    if args.dataset == 'zhang':
        args.weight_decay = 0.01

    return args


def seed_everything():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    torch.backends.cudnn.deterministic = True


def load_vocab(filepath: str):
    filepath = os.path.join('data', filepath, 'drug_list.csv')
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id


def load_feature(dataset: str):
    filepath = os.path.join('data', dataset, 'chem_Jacarrd_sim.csv')
    df = pd.read_csv(filepath, index_col=0)
    features = df.values
    return torch.tensor(features).float()


def load_csv_data(filepath: str, smiles2id: dict, is_train_file: bool = True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(filepath, index_col=False)

    edges = []
    edges_false = []
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
        else:
            continue
        if label > 0:
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
        else:
            edges_false.append((idx_1, idx_2))
            edges_false.append((idx_2, idx_1))
    if is_train_file:
        edges = np.array(edges, dtype=int)
        edges_false = np.array(edges_false, dtype=int)
        return edges, edges_false
    else:
        edges = np.array(edges, dtype=int)
        edges_false = np.array(edges_false, dtype=int)
        return edges, edges_false


def build_diffusion_graph(adj_matrix, t=1):
    """
    构建扩散图。
    :param adj_matrix: 输入图的邻接矩阵（稀疏矩阵格式）
    :param t: 扩散时间参数，控制扩散距离
    :return: 扩散图邻接矩阵（稀疏矩阵格式）
    """
    # 计算度矩阵
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-8))  # 避免除零错误

    # 计算归一化拉普拉斯矩阵
    laplacian = sp.eye(adj_matrix.shape[0]) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt

    # 计算扩散核：exp(-t * L)，L为拉普拉斯矩阵
    diffusion_matrix = sp.eye(adj_matrix.shape[0]) - t * laplacian
    diffusion_adj = diffusion_matrix.maximum(0)  # 保证非负性

    return diffusion_adj


def build_diffusion_graph_ppr(adj_matrix, alpha=0.2):
    """
    构建基于PPR的扩散图。

    :param adj_matrix: 输入图的邻接矩阵（稀疏矩阵格式）
    :param alpha: PPR的参数
    :return: 扩散图邻接矩阵（稀疏矩阵格式）
    """
    # 计算度矩阵并进行归一化
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-8))  # 避免除零错误
    norm_adj = d_inv_sqrt @ adj_matrix @ d_inv_sqrt  # 归一化邻接矩阵

    # 构建单位矩阵
    identity_matrix = sp.eye(adj_matrix.shape[0])  # 创建单位矩阵

    # 使用稀疏矩阵的逆来计算PPR矩阵
    ppr_matrix = alpha * spsolve(identity_matrix - (1 - alpha) * norm_adj, np.ones(adj_matrix.shape[0]))

    # 将PPR值转换为稀疏矩阵
    ppr_sparse = sp.diags(ppr_matrix)

    return ppr_sparse


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


def load_data(args: Namespace, smiles2idx: dict = None):
    assert smiles2idx is not None
    num_nodes = len(smiles2idx)
    train_edges, train_edges_false = load_csv_data(os.path.join('data', args.dataset, args.train_data_path), smiles2idx,
                                                   is_train_file=True)
    val_edges, val_edges_false = load_csv_data(os.path.join('data', args.dataset, args.valid_data_path), smiles2idx,
                                               is_train_file=False)
    test_edges, test_edges_false = load_csv_data(os.path.join('data', args.dataset, args.test_data_path), smiles2idx,
                                                 is_train_file=False)

    all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0)
    data = np.ones(all_edges.shape[0])

    adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])),
                        shape=(num_nodes, num_nodes))
    data_train = np.ones(train_edges.shape[0])
    data_train_false = np.ones(train_edges_false.shape[0])

    adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),
                              shape=(num_nodes, num_nodes))
    adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])), \
                                    shape=(num_nodes, num_nodes))

    return adj, adj_train, adj_train_false, val_edges, val_edges_false, test_edges, test_edges_false


args = parse_args()
logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
seed_everything()
args.cuda = True if torch.cuda.is_available() else False
# args.cuda = False
if args.cuda:
    torch.cuda.set_device(args.gpu)


def main():
    # --------------------------------------load data--------------------

    original_adj, adj_train, adj_train_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        load_data(args, smiles2idx=load_vocab(args.dataset))
    feature = load_feature(args.dataset)
    adj_diff = build_diffusion_graph_ppr(adj_train, 0.5)

    args.feature_dim = feature.shape[1]
    args.drug_nums = feature.shape[0]

    # ---------------------------log info-------------------------
    num_nodes = original_adj.shape[0]
    num_edges = original_adj.nnz

    logger.info('Dataset: {}'.format(args.dataset))
    logger.info('Number of nodes: {}, number of edges: {}'.format(num_nodes, num_edges))

    features_nonzero = 0
    args.features_nonzero = features_nonzero

    # input for model
    num_edges_w = adj_train.sum()
    num_nodes_w = adj_train.shape[0]
    args.num_edges_w = num_edges_w
    pos_weight = float(num_nodes_w ** 2 - num_edges_w) / num_edges_w

    adj_norm = normalize_adj(adj_train)
    adj_diff_norm = normalize_adj(adj_diff)

    adj_label = adj_train
    adj_mask = pos_weight * adj_train.toarray() + adj_train_false.toarray()
    adj_mask = torch.flatten(torch.Tensor(adj_mask))

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_diff_norm = sparse_mx_to_torch_sparse_tensor(adj_diff_norm)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
    lbl = torch.cat((torch.ones(num_nodes), torch.ones(num_nodes)))

    model = MSDGCL(args)
    if args.cuda:
        adj_norm = adj_norm.cuda()
        adj_diff_norm = adj_diff_norm.cuda()
        adj_label = adj_label.cuda()
        adj_mask = adj_mask.cuda()
        model = model.cuda()
        feature = feature.cuda()

    loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    best_roc, best_epoch = 0, 0

    for epoch in range(args.epochs):
        t = time.time()

        model.train()
        optimizer.zero_grad()

        preds_out, MSEloss, _ = model(feature, adj_norm, adj_diff_norm)
        preds = preds_out.view(-1)
        labels = adj_label.to_dense().view(-1)
        BCEloss = torch.mean(loss_function_BCE(preds, labels) * adj_mask)

        total_loss = BCEloss + MSEloss
        # total_loss = BCEloss

        model.eval()
        preds_out, _, _ = model(feature, adj_norm, adj_diff_norm)
        roc_curr, ap_curr, f1_curr, acc_curr = get_roc_score(
            preds_out, val_edges, val_edges_false
        )
        logger.info(
            'Epoch: {} train_loss= {:.5f} val_roc= {:.5f} val_ap= {:.5f}, val_f1= {:.5f}, val_acc={:.5f}, time= {:.5f} '.format(
                epoch + 1, total_loss, roc_curr, ap_curr, f1_curr, acc_curr, time.time() - t
            ))
        if roc_curr > best_roc and epoch > 150:
            best_roc = roc_curr
            best_epoch = epoch + 1
            if args.save_dir:
                save_checkpoint(os.path.join(args.save_dir, args.dataset, f'best_model.pt'), model, args)

        # update parameters
        total_loss.backward()
        optimizer.step()

    logger.info('Optimization Finished!')

    model = load_checkpoint(os.path.join(args.save_dir, args.dataset, 'best_model.pt'), cuda=args.cuda,
                            logger=logger)

    model.eval()
    preds_out, _, emb = model(feature, adj_norm, adj_diff_norm)
    torch.save(emb, f'{args.dataset}_drug_embeddings.pt')
    roc_score, ap_score, f1_score, acc_score = get_roc_score(
        preds_out, test_edges, test_edges_false, test=True)
    torch.save(test_edges, f'{args.dataset}_test_edges.pt')
    torch.save(test_edges_false, f'{args.dataset}_test_edges_false.pt')
    logger.info('Dataset: {}'.format(args.dataset))
    logger.info('BEST  MODEL!')
    logger.info(f'Model best_val_roc = {best_roc:.6f} on epoch {best_epoch}')
    logger.info('Test ROC score: {:.5f}'.format(roc_score))
    logger.info('Test AP score: {:.5f}'.format(ap_score))
    logger.info('Test F1 score: {:.5f}'.format(f1_score))
    logger.info('Test ACC score: {:.5f}'.format(acc_score))
    logger.info('{:.4f}\t{:.4f}\t{:.4f}'.format(roc_score, ap_score, f1_score))


if __name__ == '__main__':
    main()
