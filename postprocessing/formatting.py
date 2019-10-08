from misc import *
import numpy as np


def reshape(y_true, y_pred):
    preds_per_day = day_len // freq - ph // freq - hist // freq

    results = np.array([np.c_[y_true, y_pred]])

    return np.reshape(results, (-1, preds_per_day, 2))
