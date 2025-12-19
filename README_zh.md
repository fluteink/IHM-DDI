[English](README.md) | 中文
# LGS-DDI
"**LGS-DDI: 局部-全局协同的药物相互作用预测方法**"

## 环境依赖
```
bash
pip install -r requirements.txt
```
此外，请根据您的CUDA版本安装对应的```
torch```
版本。

## 获取药物相似性矩阵
数据集位于 `data` 目录。运行以下代码生成相似性矩阵：
**注意**：需修改 `data/jaccard_sim.py` 中的数据集路径和输出路径
```
bash
python data/jaccard_sim.py
```
## 重组相似性矩阵
生成矩阵后需进行重组：
```
bash
python data/re_generate_sim_matrix.py
```
**注意**：运行前请修改 `data/re_generate_sim_matrix.py` 中的 `similarity_matrix_file` 路径  
> ⚠️ 上述两步结果已保存在各数据集目录中，可跳过直接进入训练阶段

## 模型训练
启动训练示例：
```
bash
python main.py \
  --dataset zhang \
  --epochs 200 \
  --learning_rate 0.0005 \
  --activation ReLU \
  --dnn_dropout 0.2 \
  --hidden_dim 64 \
  --save_dir ./model_save \
  --gpu 0
```
### 参数说明
| 参数名称             | 类型   | 默认值       | 可选值/说明                                      |
|----------------------|--------|--------------|--------------------------------------------------|
| `--dataset`          | 字符串 | `DeepDDI`    | `['zhang', 'ChChMiner', 'DeepDDI']` 数据集名称   |
| `--train_data_path`  | 字符串 | `train.csv`  | 训练数据CSV路径(相对于 `data/{dataset}/`)        |
| `--valid_data_path`  | 字符串 | `valid.csv`  | 验证数据路径                                     |
| `--test_data_path`   | 字符串 | `test.csv`   | 测试数据路径                                     |
| `--learning_rate`    | 浮点数 | `0.001`      | 初始学习率                                       |
| `--epochs`           | 整数   | `300`        | 训练轮次                                         |
| `--weight_decay`     | 浮点数 | `0`          | L2正则化系数                                       |
| `--L2`               | 整数   | `0`          | (代码未使用) L2正则化标志位                      |
| `--gpu`              | 整数   | `0`          | GPU设备ID(-1表示使用CPU)                         |
| `--activation`       | 字符串 | `ELU`        | `['ReLU','LeakyReLU','PReLU','tanh','SELU','ELU']` 激活函数 |
| `--dnn_dropout`      | 浮点数 | `0.1`        | DNN层的Dropout率                                 |
| `--hidden_dim`       | 整数   | `32`         | 嵌入维度大小                                     |
| `--tau`              | 浮点数 | `0.5`        | 损失函数温度参数                                 |
| `--alpha`            | 浮点数 | `1`          | 损失权重系数                                     |
| `--save_dir`         | 字符串 | `./model_save` | 模型保存目录                                     |
| `--quiet`            | 标志位 | `False`      | 启用安静模式(--quiet)                           |

## 训练输出
- 药物嵌入文件：`{dataset}_drug_embeddings.pt`
- 测试边数据：`{dataset}_test_edges.pt` / `{dataset}_test_edges_false.pt`
- 最佳模型路径：`./model_save/{dataset}/best_model.pt`

## 可视化生成
### 2D可视化
```
bash
python visualization_2d.py
# 输出：ssdrug_pairs_2d_visualization_subplots.png (600dpi)
```
### 3D可视化

```
bash
python visualization.py
# 输出：drug_pairs_3d_visualization_subplots500.png (600dpi)
```
> ⚠️ 注意事项：
> 1. 确保安装依赖库（如`matplotlib`, `plotly`）
> 2. 输出图片为600dpi高清格式，适合论文发表
> 3. 自定义数据路径时需修改对应脚本中的路径配置

