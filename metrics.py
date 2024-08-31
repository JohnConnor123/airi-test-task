from functools import partial

import numpy as np
from scipy.stats import pearsonr as _pearsonr
from scipy.stats import rankdata
from scipy.stats import spearmanr as _spearmanr
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import ndcg_score as _ndcg_score
from sklearn.metrics import (
    precision_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)


def pearsonr(true, pred):
    return _pearsonr(true, pred)[0]


def spearmanr(true, pred):
    return _spearmanr(true, pred)[0]


def stab_spearman(true, pred, THRESHOLD):
    mask = true < THRESHOLD
    stab_spearman = _spearmanr(true[mask], pred[mask])[0]
    return stab_spearman


def ndcg_score(true, pred, K=None):
    true_ranking = rankdata(-true, method="min")
    return _ndcg_score([true_ranking], [-pred], k=K)


def det_precision_score(true, pred, THRESHOLD, K=None):
    if K is None:
        ids = np.argpartition(pred, -K)[:K]
        sample_weight = [1.0 if i in ids else 0.0 for i in range(len(true))]
    else:
        sample_weight = None
    true = true < THRESHOLD
    pred = pred < THRESHOLD
    detpr = precision_score(true, pred, sample_weight=sample_weight)
    return detpr


def compute_metrics(true, pred, THRESHOLD=-0.5, K=30):
    metrics = dict()

    regression_metrics = {
        "R2": r2_score,
        "RMSE": root_mean_squared_error,
        "Pearson": pearsonr,
        "Spearman": spearmanr,
        "StabSpearman": partial(stab_spearman, THRESHOLD=THRESHOLD),
    }

    classification_metrics = {
        "MCC": matthews_corrcoef,
        "AUC": roc_auc_score,
        "ACC": accuracy_score,
    }

    other_metrics = {
        "DetPr": partial(det_precision_score, THRESHOLD=THRESHOLD, K=K),
        "nDCG": partial(ndcg_score, K=K),
    }

    for k, metric in (regression_metrics | other_metrics).items():
        metrics[k] = metric(true, pred)

    true = true < THRESHOLD

    for k, metric in classification_metrics.items():
        # The mutation is considered stabilizing if predicted DDG < THRESHOLD=-0.5.
        # That means that the lower the prediction of the model the more
        # mutation is likely to be stabilizing.
        # Hence, to correctly calculate AUC score we must invert the predictions of the model:
        pred_ = pred < THRESHOLD if k != "AUC" else -pred
        try:
            metrics[k] = metric(true, pred_)
        except Exception:
            metrics[k] = 0.0

    return metrics
