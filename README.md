English | [中文](README_zh.md)

# LGS-DDI

The implementation of "**LGS-DDI: Local-Global Synergy for Drug-Drug Interaction Prediction**"

## Requirements
```bash
pip install -r requirements.txt
```
In addition to these pacages, please install ```torch``` according to your CUDA version.

## Obtain drug similarity matrix
Our dataset is in the `data` directory. Please run the following code to generate the similarity matrix for each dataset. **Note**: You need to modify the dataset path in `data/jaccard_sim.py` and the output path for the similarity matrix.
```bash
python data/jaccard_sim.py
```
## Reorganize the similarity matrix
After generating the similarity matrix, we need to reorganize it. Please run the following code to reorder the similarity matrix. **Note**: Before running, modify the path of `similarity_matrix_file` in `data/re_generate_sim_matrix.py`.
```bash
python data/re_generate_sim_matrix.py
```
**Note:** The results of the above two steps have already been saved in each dataset's corresponding directory. **You may skip re-running the previous code** and proceed directly to the next step.  
## Model Training
To train the model, please run the following code.
```bash
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
Here is a description of the parameters used in the code:

| Parameter Name        | Type  | Default Value  | Optional Values / Description                                |
|-----------------------| ----- | -------------- | ------------------------------------------------------------ |
| `--dataset`           | str   | `DeepDDI`      | `['zhang', 'ChChMiner', 'DeepDDI']` Dataset name             |
| `--train_data_path`   | str   | `train.csv`    | Path to training data CSV file (relative to `data/{dataset}/`) |
| `--valid_data_path`   | str   | `valid.csv`    | Path to validation data CSV file                             |
| `--test_data_path`    | str   | `test.csv`     | Path to test data CSV file                                   |
| `--learning_rate`     | float | `0.001`        | Initial learning rate                                        |
| `--epochs`            | int   | `300`          | Number of training epochs                                    |
| `--weight_decay`      | float | `0`            | Weight decay (L2 regularization coefficient)                 |
| `--L2`                | int   | `0`            | (Unused in code) Placeholder for L2 regularization flag      |
| `--gpu`               | int   | `0`            | GPU device ID (set to `-1` to force CPU usage)               |
| `--activation`        | str   | `ELU`          | `['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']` Activation function |
| `--dnn_dropout`       | float | `0.1`          | Dropout rate for DNN layers                                  |
| `--hidden_dim`        | int   | `32`           | Embedding dimension size                                     |
| `--tau`               | float | `0.5`          | Temperature parameter for loss function                      |
| `--alpha`             | float | `1`            | Loss weight coefficient                                      |
| `--save_dir`          | str   | `./model_save` | Directory for saving model checkpoints                       |
| `--quiet`             | flag  | `False`        | Skip non-essential log messages (add `--quiet` to enable)    |
## Training Outputs:
- Drug embeddings file: `{dataset}_drug_embeddings.pt`
- Test edge data: `{dataset}_test_edges.pt` / `{dataset}_test_edges_false.pt`
- Best model save path: `./model_save/{dataset}/best_model.pt`

## Visualization Generation  
### 2D Visualization  
```bash  
python visualization_2d.py  
# Output: ssdrug_pairs_2d_visualization_subplots.png (600dpi)  
```

### 3D Visualization  
```bash  
python visualization.py  
# Output: drug_pairs_3d_visualization_subplots500.png (600dpi)  
```

**Notes:**  
- Ensure Python environment and dependencies (e.g., `matplotlib`, `plotly`) are installed.  
- The scripts generate high-resolution (600dpi) images for publication-quality figures.  
- Verify input data paths in the scripts if customized.