from models.tools.fcn_submodules import *
import numpy as np
import torch.nn as nn
import torch
import os
from misc import path
from torch.utils.data import TensorDataset
from torch import Tensor
from training.pytorch.training import fit, predict
from datetime import datetime
from misc import hist_freq
from models.FCN import FCN
from models.tools.files import *

params = {
    "n_in": 3,

    "encoder_channels": [64, 128, 64],
    "encoder_kernel_sizes": [3, 3, 3],
    "encoder_dropout": 0.5,

    "decoder_channels": [2048],
    "decoder_dropout": 0.5,

    "epochs": 500,
    "bs": 100,
    "lr": 1e-4,
    "patience": 250,

    "l2": 0.0,  # 1e-2,

    "lambda": 0.17,
    "n_domains": 4,

    "checkpoint": None,
}


class DAFCN():
    def __init__(self, params):
        self.params = params

        self.model = DANN_FCN_Module(n_in=self.params["n_in"],
                                     encoder_channels=self.params["encoder_channels"],
                                     encoder_kernel_sizes=self.params["encoder_kernel_sizes"],
                                     encoder_dropout=self.params["encoder_dropout"],
                                     decoder_channels=self.params["decoder_channels"],
                                     decoder_dropout=self.params["decoder_dropout"],
                                     n_domains=self.params["n_domains"])

        self.model.cuda()

        self._create_and_load_checkpoint(self.params["checkpoint"])

        self.loss = DALoss(lambda_=self.params["lambda"])

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

    def fit(self, train_x, train_y, valid_x, valid_y):
        train_ds = self._reshape_and_create_tensor(train_x, train_y)
        valid_ds = self._reshape_and_create_tensor(valid_x, valid_y)

        fit(epochs=self.params["epochs"],
            batch_size=self.params["bs"],
            model=self.model,
            loss_func=self.loss,
            opt=self.opt,
            train_ds=train_ds,
            valid_ds=valid_ds,
            patience=self.params["patience"],
            checkpoint_file=self.checkpoint_file)

    def predict(self, x, y, file=None):
        if file is None:
            self.model.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.model.load_state_dict(torch.load(file))

        ds = self._reshape_and_create_tensor(x, y)

        [y_trues_glucose, y_trues_subject], [y_preds_glucose, y_preds_subject] = predict(self.model, ds)

        y_trues_glucose, y_preds_glucose = [_.reshape(-1, 1) for _ in [y_trues_glucose, y_preds_glucose]]
        y_trues_subject = y_trues_subject.reshape(-1, 1)
        y_preds_subject = np.argmax(y_preds_subject, axis=1).reshape(-1, 1)

        return np.c_[y_trues_glucose, y_trues_subject], np.c_[y_preds_glucose, y_preds_subject]

    def extract_features(self, x, y):
        x, _ = self._reshape_x_y(x, y)

        self.model.eval()

        features = self.model.encoder(Tensor(x).cuda()).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]

    def load_weights_from_file(self, file_name):
        path = compute_weights_path(file_name)
        self.model.load_state_dict(torch.load(path))
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def save_weights(self, file_name):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        torch.save(self.model.state_dict(), compute_checkpoint_path(file_name))

    def save_encoder_regressor(self, file_name):
        """ Save a FCN model based on the encoder/regressor part of DAFCN """
        fcn_params = params.copy()
        del fcn_params["lambda"]
        del fcn_params["n_domains"]

        fcn = FCN(fcn_params)
        fcn.load_weights(self.model.encoder.state_dict(), self.model.regressor.state_dict())

        file_name = file_name.split(os.sep)
        file_name[-1] = fcn.__class__.__name__ + "_" + file_name[-1] + ".pt"
        file_name = os.path.join(*file_name)

        fcn.save_weights(file_name)

    def _create_and_load_checkpoint(self, checkpoint):
        if checkpoint is None:
            self.checkpoint_file = compute_checkpoint_path(compute_checkpoint_file_name(self))
        else:
            self.checkpoint_file = compute_checkpoint_path(params["checkpoint"])
            self.model.load_state_dict(torch.load(self.checkpoint_file))

    def _reshape_x_y(self, x, y):
        x = x.reshape(x.shape[0], -1, 3)
        x = x.transpose(0, 2, 1)
        y = np.expand_dims(y, axis=2)
        return x, y

    def _reshape_and_create_tensor(self, x, y):
        x, y = self._reshape_x_y(x, y)
        return TensorDataset(Tensor(x).cuda(), Tensor(y).cuda())

    def clear_checkpoint(self):
        os.remove(self.checkpoint_file)


class DANN_FCN_Module(nn.Module):
    def __init__(self, n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout, decoder_channels,
                 decoder_dropout, n_domains):
        super(DANN_FCN_Module, self).__init__()

        decoder_input_dims = [encoder_channels[-1]]
        decoder_kernel_sizes = [compute_decoder_kernel_size(encoder_kernel_sizes)]

        self.encoder = FCN_Encoder_Module(n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout)
        self.regressor = FCN_Regressor_Module(decoder_input_dims, decoder_channels, decoder_kernel_sizes,
                                              decoder_dropout)
        self.domain_classifier = FCN_Domain_Classifier_Module(decoder_input_dims, decoder_channels,
                                                              decoder_kernel_sizes, decoder_dropout, n_domains)

    def forward(self, input):
        features = self.encoder(input)
        prediction = self.regressor(features)
        domain = self.domain_classifier(features)

        return prediction, domain


class DALoss(nn.Module):
    def __init__(self, lambda_):
        super(DALoss, self).__init__()

        self.lambda_ = lambda_

        self.mse = nn.MSELoss()
        self.nll = nn.NLLLoss()

    def forward(self, x, y):
        y_preds = x[0].squeeze(2)
        domain_preds = x[1].squeeze(2)

        y_trues = y[:, 0]
        domain_trues = y[:, 1].long().squeeze(1)

        mse = self.mse(y_preds, y_trues)
        nll = self.nll(domain_preds, domain_trues)

        return mse + self.lambda_ * nll, mse, nll
