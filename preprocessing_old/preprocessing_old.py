import numpy as np
import pandas as pd
from _misc import *
from preprocessing_old.files_processing import subject_files, _remove_subject_from_pool, _get_subject_file_from_name
from preprocessing_old.formatting import _reshape_data_into_days, _merge_days, _merge_pool_subjects, \
    _per_subject_merge_days, _create_samples, _dann_add_domains, _x_y_split
from preprocessing_old.cross_validation import _per_subject_split, _split_train_valid, _split_train_valid_test
from preprocessing_old.data import Data
from preprocessing_old.scaling import _per_subject_standardization, _standardize_pool


def preprocessing(source_dataset, target_dataset, target_subject):
    pool_data = source_preprocessing(target_dataset, target_subject, source_dataset)
    subject_data = target_preprocessing(target_dataset, target_subject)

    return pool_data, subject_data


def source_preprocessing(source_dataset, target_dataset, target_subject):

    pool_files = _remove_subject_from_pool(source_dataset, target_dataset, target_subject)
    data = [_global_preprocessing(file) for file in pool_files]
    train, valid = _per_subject_split(data, cv, _split_train_valid)
    train, valid = [_per_subject_merge_days(set) for set in [train, valid]]
    [train, valid], mean, std  = _per_subject_standardization(train, valid)

    # for DANN, add domain for subject before merging
    train, valid = [_dann_add_domains(set) for set in [train, valid]]

    train, valid = _merge_pool_subjects([train, valid])
    train, valid = [_x_y_split(_, index=-2) for _ in [train, valid]]
    return Data(train, valid, valid.copy(), mean.copy(), std.copy())

def target_preprocessing(target_dataset, target_subject):
    file = _get_subject_file_from_name(target_dataset, target_subject)
    data = _global_preprocessing(file)
    train, valid, test = _split_train_valid_test(data, cv)
    train, valid, test = _merge_days(train), _merge_days(valid), _merge_days(test)
    [train, valid, test], mean, std = _standardize_pool(train, [valid, test])
    train, valid, test = [_x_y_split(_, index=-1) for _ in [train, valid, test]]
    return Data(train, valid, test, mean, std)


def _global_preprocessing(file):
    data = _load_data(file)
    data = _reshape_data_into_days(data)
    data = _create_samples(data)
    return data


def _load_data(file):
    return pd.read_csv(file)
