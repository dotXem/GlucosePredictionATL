from postprocessing_old.formatting import reshape
from postprocessing_old.rescaling import rescale
import numpy as np


def source_postprocessing(y_true, y_pred, mean, std):
    y_true_glucose = np.concatenate(
        [rescale(y_true[y_true[:, 1] == i, 0], mean[i], std[i]) for i in range(int(max(y_true[:, 1])+1))], axis=0)
    y_pred_glucose = np.concatenate(
        [rescale(y_pred[y_true[:, 1] == i, 0], mean[i], std[i]) for i in range(int(max(y_true[:, 1])+1))], axis=0)

    return reshape(y_true_glucose, y_pred_glucose), reshape(y_true[:, 1], y_pred[:, 1])


def target_postprocessing(y_true, y_pred, mean, std):
    y_true = rescale(y_true, mean, std)
    y_pred = rescale(y_pred, mean, std)

    return reshape(y_true, y_pred)
