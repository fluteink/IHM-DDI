import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 数据集列表
datasets = ['zhang', 'ChChMiner', 'DeepDDI']
titles = ['ZhangDDI', 'ChChMiner', 'DeepDDI']

# 创建画布和子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列的子图布局

# 遍历数据集
for i, dataset in enumerate(datasets):
    # 加载数据
    emb = torch.load(f'{dataset}_drug_embeddings.pt', weights_only=False)  # 全部嵌入
    test_edges = torch.load(f'{dataset}_test_edges.pt', weights_only=False)  # 正样本药物对
    test_edges_false = torch.load(f'{dataset}_test_edges_false.pt', weights_only=False)  # 负样本药物对

    # 从正样本中随机选择 1000 个样本
    num_samples = 1000
    if len(test_edges) > num_samples:
        sampled_indices = np.random.choice(len(test_edges), num_samples, replace=False)
        test_edges_balanced = test_edges[sampled_indices]
    else:
        test_edges_balanced = test_edges  # 如果正样本不足，直接使用所有正样本

    # 从负样本中随机选择 1000 个样本
    if len(test_edges_false) > num_samples:
        sampled_indices = np.random.choice(len(test_edges_false), num_samples, replace=False)
        test_edges_false_balanced = test_edges_false[sampled_indices]
    else:
        test_edges_false_balanced = test_edges_false  # 如果负样本不足，直接使用所有负样本


    # 使用哈达玛积生成药物对的嵌入
    def get_drug_pair_embedding(emb, drug_pairs):
        """
        生成药物对的嵌入（哈达玛积）
        :param emb: 单个药物的嵌入 (num_drugs, embedding_dim)
        :param drug_pairs: 药物对列表，每个元素是 (drug1_id, drug2_id)
        :return: 药物对的嵌入 (num_pairs, embedding_dim)
        """
        pair_embeddings = []
        for drug1_id, drug2_id in drug_pairs:
            drug1_emb = emb[drug1_id]  # 获取药物1的嵌入
            drug2_emb = emb[drug2_id]  # 获取药物2的嵌入
            pair_emb = drug1_emb * drug2_emb  # 哈达玛积
            pair_embeddings.append(pair_emb)
        return torch.stack(pair_embeddings)


    # 生成正样本和负样本药物对的嵌入
    pair_emb_pos = get_drug_pair_embedding(emb, test_edges_balanced)
    pair_emb_neg = get_drug_pair_embedding(emb, test_edges_false_balanced)

    # 将正样本和负样本药物对的嵌入拼接在一起
    pair_emb_all = torch.cat([pair_emb_pos, pair_emb_neg], dim=0)

    # 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=num_samples, learning_rate=50, n_iter=5000)
    pair_emb_2d = tsne.fit_transform(pair_emb_all.detach().cpu().numpy())

    # 分离正样本和负样本的 2D 嵌入
    pair_emb_pos_2d = pair_emb_2d[:num_samples]
    pair_emb_neg_2d = pair_emb_2d[num_samples:]

    # 在对应子图中绘制可视化结果
    ax = axes[i]
    ax.scatter(pair_emb_pos_2d[:, 0], pair_emb_pos_2d[:, 1], c='#1f77b4', label='Positive Pairs', alpha=0.5)
    ax.scatter(pair_emb_neg_2d[:, 0], pair_emb_neg_2d[:, 1], c='#ff7f0e', label='Negative Pairs', alpha=0.5)

    # 设置子图标题
    ax.set_title(titles[i], fontsize=14, pad=15)
    # ax.set_xlabel()
    # ax.set_ylabel()

    # 只在最后一个子图中显示图例
    if i == len(datasets) - 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.43, 1.1), fontsize=12)

# 调整布局并保存为 600dpi 的图片
plt.tight_layout()
plt.savefig('ssdrug_pairs_2d_visualization_subplots.png', dpi=600, bbox_inches='tight', format='png')
plt.show()
