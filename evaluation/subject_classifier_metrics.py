import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def precision(y_true, y_pred):
    y_true, y_pred = [np.concatenate(_) for _ in [y_true, y_pred]]
    return precision_score(y_true, y_pred, average="macro")

def recall(y_true, y_pred):
    y_true, y_pred = [np.concatenate(_) for _ in [y_true, y_pred]]
    return recall_score(y_true, y_pred, average="macro")

def f1(y_true, y_pred):
    y_true, y_pred = [np.concatenate(_) for _ in [y_true, y_pred]]
    return f1_score(y_true, y_pred, average="macro")