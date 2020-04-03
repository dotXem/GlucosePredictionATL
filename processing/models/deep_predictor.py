from torch.utils.data import TensorDataset
import os
from misc.utils import printd
import sys
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch

# from _old.models_old.tools.files import compute_checkpoint_file_name, compute_checkpoint_path


class DeepPredictor(ABC):
    def __init__(self, subject, ph, params, train, valid, test):
        self.subject = subject
        self.params = params
        self.ph = ph

        self.train_x, self.train_y, self.train_t = self._reshape(train)
        self.valid_x, self.valid_y, self.valid_t = self._reshape(valid)
        self.test_x, self.test_y, self.test_t = self._reshape(test)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    #TODO rework description
    def _reshape(self, data):
        t = data["datetime"]
        y = data["y"]
        x = data.drop(["y","datetime"], axis=1)

        return x, y, t

    def _str2dataset(self, dataset_name):
        if dataset_name in ["train", "training_old"]:
            return self.train_x, self.train_y, self.train_t
        elif dataset_name in ["valid", "validation"]:
            return self.valid_x, self.valid_y, self.valid_t
        elif dataset_name in ["test", "testing"]:
            return self.test_x, self.test_y, self.test_t
        else:
            printd("Dataset name not known.")
            sys.exit(-1)

    def _format_results(self, y_true, y_pred, t):
        return pd.DataFrame(data=np.c_[y_true,y_pred],index=pd.DatetimeIndex(t.values),columns=["y_true", "y_pred"])

    def load_weights_from_file(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def _format_results_source(self, y_true, y_pred, t):
        return pd.DataFrame(data=np.c_[y_true,y_pred],index=pd.DatetimeIndex(t.values),columns=["y_true", "d_true", "y_pred", "d_pred"])


    def to_dataset(self, x, y):
        return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())

    def clear_checkpoint(self):
        os.remove(self.checkpoint_file)