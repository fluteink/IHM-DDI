import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 使用GAT模型
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling

# 1. 读取数据
# 药物相互作用数据
interaction_df = pd.read_csv('train.csv')  # 格式: drugbank_id_1,drugbank_id_2,smiles_1,smiles_2,label
# 相似度矩阵
similarity_df = pd.read_csv(r"D:\研究生\论文\DDI\小论文\ThreeSection\data\ChChMiner\chem_Jacarrd_sim.csv", index_col=0)  # 第一列和第一行是drugbank_id

# 2. 创建药物ID到索引的映射
all_drugs = set(interaction_df['drugbank_id_1']).union(set(interaction_df['drugbank_id_2']))
drug_to_idx = {drug: idx for idx, drug in enumerate(sorted(all_drugs))}
num_drugs = len(drug_to_idx)

# 3. 处理相似度矩阵
# 确保相似度矩阵中的药物都在我们的映射中
similarity_matrix = np.zeros((num_drugs, num_drugs))
for i, drug1 in enumerate(similarity_df.index):
    if drug1 in drug_to_idx:
        idx1 = drug_to_idx[drug1]
        for j, drug2 in enumerate(similarity_df.columns):
            if drug2 in drug_to_idx:
                idx2 = drug_to_idx[drug2]
                similarity_matrix[idx1, idx2] = similarity_df.iloc[i, j]


# 4. 从SMILES生成分子指纹作为节点特征
def smiles_to_fingerprint(smiles, size=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size))
    else:
        return [0] * size


# 创建所有药物的特征矩阵
drug_features = {}
for _, row in interaction_df.iterrows():
    for drug_id, smiles in [(row['drugbank_id_1'], row['smiles_1']),
                            (row['drugbank_id_2'], row['smiles_2'])]:
        if drug_id not in drug_features:
            drug_features[drug_id] = smiles_to_fingerprint(smiles)

# 确保所有药物都有特征
x = torch.tensor([drug_features[drug] for drug in sorted(drug_to_idx.keys())], dtype=torch.float)

# 5. 构建图数据
# 使用相似度矩阵创建边 (只保留相似度高于阈值的边)
threshold = 0.5  # 可调整
edge_index = []
edge_attr = []

for i in range(num_drugs):
    for j in range(i + 1, num_drugs):  # 避免重复
        if similarity_matrix[i, j] > threshold:
            edge_index.append([i, j])
            edge_attr.append(similarity_matrix[i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

# 6. 准备训练数据 (药物对和标签)
# 将药物对转换为索引对
pairs = []
labels = []
for _, row in interaction_df.iterrows():
    drug1 = row['drugbank_id_1']
    drug2 = row['drugbank_id_2']
    label = row['label']
    pairs.append([drug_to_idx[drug1], drug_to_idx[drug2]])
    labels.append(label)

pairs = torch.tensor(pairs, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)

# 划分训练集和测试集
train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
train_pairs = pairs[train_idx]
train_labels = labels[train_idx]
test_pairs = pairs[test_idx]
test_labels = labels[test_idx]


# 7. 定义图注意力网络模型
class DrugInteractionGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(DrugInteractionGAT, self).__init__()
        # edge_dim 参数告诉 GATConv 边特征的维度
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, edge_dim=1, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=1, dropout=0.2)
        self.lin = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, edge_attr, pair_indices):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        drug1_features = x[pair_indices[:, 0]]
        drug2_features = x[pair_indices[:, 1]]
        pair_features = torch.cat([drug1_features, drug2_features], dim=1)
        return torch.sigmoid(self.lin(pair_features)).view(-1)


# 7. 定义图卷积网络模型
class DrugInteractionGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(DrugInteractionGCN, self).__init__()
        from torch_geometric.nn import GCNConv  # 使用GCN模型

        self.conv1 = GCNConv(num_features, hidden_dim)  # 不需要 edge_dim 参数
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, edge_attr, pair_indices):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        drug1_features = x[pair_indices[:, 0]]
        drug2_features = x[pair_indices[:, 1]]
        pair_features = torch.cat([drug1_features, drug2_features], dim=1)
        return torch.sigmoid(self.lin(pair_features)).view(-1)


# 8. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrugInteractionGCN(num_features=x.size(1), hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# 准备图数据
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


def train():
    model.train()
    optimizer.zero_grad()

    # 正样本
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, train_pairs.to(device))
    loss = criterion(out, train_labels.to(device))

    # 负采样
    neg_pairs = negative_sampling(edge_index, num_nodes=num_drugs,
                                  num_neg_samples=train_pairs.size(0)).t().to(device)
    neg_out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, neg_pairs)
    neg_loss = criterion(neg_out, torch.zeros_like(neg_out))

    total_loss = loss + neg_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def tes(pairs, labels):
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, pairs.to(device))
        pred = (out > 0.5).float()
        correct = (pred == labels.to(device)).sum().item()
        acc = correct / len(labels)
    return acc


# 训练循环
for epoch in range(1, 1010):
    loss = train()
    if epoch % 10 == 0:
        train_acc = tes(train_pairs, train_labels)
        test_acc = tes(test_pairs, test_labels)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 9. 保存模型
torch.save(model.state_dict(), 'drug_interaction_model.pth')