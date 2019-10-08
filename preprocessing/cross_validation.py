import sklearn.model_selection as sk_model_selection
import numpy as np
from misc import seed

def _split_train_valid(data, cv):
    train, valid, test = [], [], []

    kf = sk_model_selection.KFold(n_splits=cv, shuffle=True, random_state=seed)
    for train_index, valid_index in kf.split(np.arange(len(data))):
        train_fold = [data[i].copy() for i in train_index]
        valid_fold = [data[i].copy() for i in valid_index]
        train.append(train_fold)
        valid.append(valid_fold)

    return train, valid


def _split_train_valid_test(data, cv):
    train, valid, test = [], [], []

    kf_1 = sk_model_selection.KFold(n_splits=cv, shuffle=True, random_state=seed)
    for train_valid_index, test_index in kf_1.split(np.arange(len(data))):
        train_valid_fold = [data[i] for i in train_valid_index]
        test_fold = [data[i].copy() for i in test_index]
        kf_2 = sk_model_selection.KFold(n_splits=cv - 1, shuffle=False)  # we already shuffled once
        for train_index, valid_index in kf_2.split(train_valid_fold):
            train_fold = [train_valid_fold[i].copy() for i in train_index]
            valid_fold = [train_valid_fold[i].copy() for i in valid_index]
            train.append(train_fold)
            valid.append(valid_fold)
            test.append(test_fold)

    return train, valid, test


def _per_subject_split(data, cv, split_func):
    data = [split_func(sbj_data, cv) for sbj_data in data]
    data = np.transpose(data, (1, 2, 0))  # (sbj, set, split) -> (set, split, sbj)
    return data