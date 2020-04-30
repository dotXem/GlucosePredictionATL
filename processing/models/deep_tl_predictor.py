import torch.nn as nn
from torch.utils.data import TensorDataset
import os
import pandas as pd
import numpy as np
import torch
from processing.models.predictor import Predictor
from misc.utils import printd
import misc.constants as cs
from .pytorch_tools.losses import DALoss
from torch import Tensor

class DeepTLPredictor(Predictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)
        self.checkpoint_file = self._compute_checkpoint_file(self.__class__.__name__)
        self.n_domains = self._compute_number_of_domains()
        self.domain_weights = self._compute_domain_weights()
        self.input_shape = self._compute_input_shape()

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def _format_results_source(self, y_trues_glucose, y_trues_subject, y_preds_glucose, y_preds_subject, t):
        y_trues_glucose, y_preds_glucose = [_.reshape(-1, 1) for _ in [y_trues_glucose, y_preds_glucose]]
        y_trues_subject = y_trues_subject.reshape(-1, 1)
        y_preds_subject = np.argmax(y_preds_subject, axis=1).reshape(-1, 1)
        y_true, y_pred = np.c_[y_trues_glucose, y_trues_subject], np.c_[y_preds_glucose, y_preds_subject]
        return pd.DataFrame(data=np.c_[y_true,y_pred],index=pd.DatetimeIndex(t.values),columns=["y_true", "d_true", "y_pred", "d_pred"])

    def _to_tensor_ds(self, x, y):
        return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())

    def _clear_checkpoint(self):
        os.remove(self.checkpoint_file)

    def _compute_checkpoint_file(self, model_name):
        rnd = np.random.randint(1e7)
        checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", model_name + "_" + str(rnd) + ".pt")
        printd("Saved model's file:", checkpoint_file)
        return checkpoint_file

    def _compute_number_of_domains(self):
        if self.params["domain_adversarial"]:
            _, y_train, _ = self._str2dataset("train")
            return int(np.max(y_train[:, 1]) + 1)
        else:
            return 0

    def _compute_domain_weights(self):
        _, y_train, _ = self._str2dataset("train")
        if self.params["domain_weights"]:
            n_samples_by_domain = [np.sum(y_train[:, 1] == i) / len(y_train) for i in
                                   range(int(max(y_train[:, 1])) + 1)]
            domains_weights = Tensor(n_samples_by_domain)
            domains_weights = 1 / domains_weights
            domains_weights /= domains_weights.min()
            domains_weights = domains_weights.cuda()
        else:
            domains_weights = None

        return domains_weights


    def _compute_loss_func(self):
        if self.params["domain_adversarial"]:
            loss_func = DALoss(self.params["da_lambda"], self.domain_weights)
        else:
            loss_func = nn.MSELoss()

        return loss_func

    def _reshape(self, data):
        t = data["datetime"]
        if self.params["domain_adversarial"]:
            y = data[["y", "domain"]].values
        else:
            y = data["y"].values

        g = data.loc[:, [col for col in data.columns if "glucose" in col]].values
        cho = data.loc[:, [col for col in data.columns if "CHO" in col]].values
        ins = data.loc[:, [col for col in data.columns if "insulin" in col]].values

        # reshape timeseties in (n_samples, hist, 1) shape and concatenate
        g = g.reshape(-1, g.shape[1], 1)
        cho = cho.reshape(-1, g.shape[1], 1)
        ins = ins.reshape(-1, g.shape[1], 1)
        x = np.concatenate([g, cho, ins], axis=2)

        return x, y, t

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)
        self.model.load_state_dict(torch.load(file))
        self.model.eval()
        features = self.model.encoder(Tensor(x).cuda()).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]