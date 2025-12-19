import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE

# 数据集列表
datasets = ['zhang', 'ChChMiner', 'DeepDDI']


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
        # pair_emb = drug1_emb * drug2_emb  # 哈达玛积
        pair_emb = drug1_emb * drug1_emb + drug2_emb * drug2_emb  # 哈达玛积
        pair_embeddings.append(pair_emb)
    return torch.stack(pair_embeddings)


# 生成 3D 可视化图
def visualize_3d_matplotlib(ax, pos_3d, neg_3d, title, show_legend=False):
    """
    生成 3D 可视化图
    :param ax: 子图的轴对象
    :param pos_3d: 正样本的 3D 嵌入
    :param neg_3d: 负样本的 3D 嵌入
    :param title: 子图的标题
    :param show_legend: 是否显示图例
    """
    if title == 'zhang':
        title = 'ZhangDDI'
    # 设置视角
    ax.view_init(elev=5,azim=90)  # 仰角 5°，方位角 135°

    # 绘制正样本和负样本
    ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c='#1f77b4', label='Positive Pairs', alpha=0.8, s=50)
    ax.scatter(neg_3d[:, 0], neg_3d[:, 1], neg_3d[:, 2], c='#ff7f0e', label='Negative Pairs', alpha=0.8, s=50)

    # 设置标题
    ax.set_title(title, color='#333333', fontsize=14, pad=15)  # pad 控制标题与子图的距离

    # 去除 X、Y、Z 轴标签
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.set_xlabel()
    # ax.set_ylabel()
    # ax.set_zlabel()

    # 设置背景和网格线颜色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#D3D3D3')
    ax.yaxis.pane.set_edgecolor('#D3D3D3')
    ax.zaxis.pane.set_edgecolor('#D3D3D3')
    ax.grid(color='#D3D3D3', linestyle='--', linewidth=0.5)

    # 只在最后一个子图中显示图例
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=12)


# 创建画布和子图
fig = plt.figure(figsize=(18, 6))  # 画布大小为 18x6
axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]  # 创建 3 个子图

# 遍历数据集
for i, dataset in enumerate(datasets):
    # 加载数据
    emb = torch.load(f'{dataset}_drug_embeddings.pt', weights_only=False)  # 全部嵌入
    test_edges = torch.load(f'{dataset}_test_edges.pt', weights_only=False)  # 正样本药物对
    test_edges_false = torch.load(f'{dataset}_test_edges_false.pt', weights_only=False)  # 负样本药物对

    # 从正样本中随机选择800个样本
    num_samples = 500
    if len(test_edges) > num_samples:
        sampled_indices = np.random.choice(len(test_edges), num_samples, replace=False)
        test_edges_balanced = test_edges[sampled_indices]
    else:
        test_edges_balanced = test_edges  # 如果正样本不足，直接使用所有正样本

    # 从负样本中随机选择800个样本
    if len(test_edges_false) > num_samples:
        sampled_indices = np.random.choice(len(test_edges_false), num_samples, replace=False)
        test_edges_false_balanced = test_edges_false[sampled_indices]
    else:
        test_edges_false_balanced = test_edges_false  # 如果负样本不足，直接使用所有负样本

    # 生成正样本和负样本药物对的嵌入
    pair_emb_pos = get_drug_pair_embedding(emb, test_edges_balanced)
    pair_emb_neg = get_drug_pair_embedding(emb, test_edges_false_balanced)

    # 将正样本和负样本药物对的嵌入拼接在一起
    pair_emb_all = torch.cat([pair_emb_pos, pair_emb_neg], dim=0)

    # 使用 t-SNE 降维到 3D
    tsne = TSNE(n_components=3, perplexity=num_samples, n_iter=5000)
    # tsne = TSNE(n_components=3)
    pair_emb_3d = tsne.fit_transform(pair_emb_all.detach().cpu().numpy())

    # 分离正样本和负样本的 3D 嵌入
    pair_emb_pos_3d = pair_emb_3d[:num_samples]
    pair_emb_neg_3d = pair_emb_3d[num_samples:]

    # 在对应子图中绘制可视化结果
    show_legend = (i == len(datasets) - 1)  # 只在最后一个子图中显示图例
    visualize_3d_matplotlib(axes[i], pair_emb_pos_3d, pair_emb_neg_3d, dataset, show_legend)

# 调整布局并保存为 600dpi 的图片
plt.tight_layout()
plt.show()
plt.savefig('drug_pairs_3d_visualization_subplots500.png', dpi=600, bbox_inches='tight', format='png')
