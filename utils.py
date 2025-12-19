import logging
import os
from argparse import Namespace
from typing import Tuple

from typing import Union, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve

from model import MSDGCL


def save_checkpoint(path: str,
                    model,
                    args: Namespace = None):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None,
                    ddi: bool = False, model_type='final'):
    debug = logger.debug if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    args.cuda = cuda if cuda is not None else args.cuda
    # Create model and optimizer
    model = MSDGCL(args)

    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability. 

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
    return accuracy_score(targets, hard_preds)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def gen_preds(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    return preds, preds_neg


def eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2 * i] > 0.95 and preds_all[2 * i + 1] > 0.95:
            preds_all[2 * i] = max(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
        else:
            preds_all[2 * i] = min(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
    # 检查是否有 NaN 值
    preds_all = np.nan_to_num(preds_all)
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >= optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_



def get_roc_score(
        rec,
        edges_pos: np.ndarray,
        edges_neg: Union[np.ndarray, List[list]],
        test=None) -> Tuple[float, float]:


    rec = rec.detach().cpu().numpy()
    adj_rec = rec

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_score_ = f1_score(labels_all, preds_all_)
    acc_score = accuracy_score(labels_all, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score





