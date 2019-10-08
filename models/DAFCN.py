from torch.autograd import Function
import numpy as np
import torch.nn as nn
import torch
import os
from misc import path, ph, hist
from torch.utils.data import TensorDataset
from torch import Tensor
from tools.pytorch.training import fit, predict
from datetime import datetime
from misc import hist_freq

params = {
    "n_in": 3,

    "encoder_channels": [64, 128, 64],
    "encoder_kernel_sizes": [3, 3, 3],
    "encoder_dropout": 0.5,

    "decoder_channels": [2048],
    "decoder_dropout": 0.5,

    # training
    "epochs": 0,
    "bs": 100,
    "lr": 1e-4, #1e-3
    "patience": 50,

    "l2": 0.0,#1e-2,

    "lambda": 0.05,
    "n_domains": 4,

    "checkpoint": os.path.join("lambda0_noL2_lr4","full_IDIAB1_0.pt"),
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

        self.loss = DANN_Loss(lambda_=self.params["lambda"])

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
        # TODO check rework with predict in training.py
        if file is None:
            self.model.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.model.load_state_dict(torch.load(file))

        ds = self._reshape_and_create_tensor(x, y)

        [y_trues_glucose, y_trues_subject], [y_preds_glucose, y_preds_subject] = predict(self.model, ds)

        y_trues_glucose, y_preds_glucose = [_.reshape(-1,1) for _ in [y_trues_glucose, y_preds_glucose]]
        y_trues_subject = y_trues_subject.reshape(-1,1)
        y_preds_subject = np.argmax(y_preds_subject,axis=1).reshape(-1,1)


        # y_trues, y_preds = y_trues[0][:,0], y_preds[0]
        #
        # return y_trues.reshape(-1, 1), y_preds.reshape(-1, 1)

        return np.c_[y_trues_glucose, y_trues_subject], np.c_[y_preds_glucose, y_preds_subject]

    def extract_features(self, x, y):
        x, _ = self._reshape_x_y(x, y)

        # x = x.reshape(x.shape[0], -1, 3)
        # x = x.transpose(0, 2, 1)

        self.model.eval()
        features = self.model.encoder(Tensor(x).cuda()).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]

    def freeze_features_extractor(self):
        for param in self.model.encoder.parameters():
            param.requires_grad_(False)

        self.model.encoder.eval()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

    def load_model(self, full_file_name):
        self.model.load_state_dict(torch.load(full_file_name))

    def save_model(self, file_name):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        torch.save(self.model.state_dict(), self._compute_checkpoint_path(file_name))

    def load_features_extractor(self, full_file_path):
        self.model.encoder.load_state_dict(torch.load(full_file_path))

    def save_encoder_weights(self, file_name):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        torch.save(self.model.encoder.state_dict(), self._compute_checkpoint_path(file_name))

    def _compute_checkpoint_path(self, file_name):
        return os.path.join(path, "tmp", "checkpoints", file_name + ".pt")

    def _create_and_load_checkpoint(self, checkpoint):
        if checkpoint is None:
            name = str(datetime.now().strftime("%Y_%m_%d_%H_%M")) + "dann_fcn" + str(np.random.randint(100))
            self.checkpoint_file = self._compute_checkpoint_path(name)
        else:
            # self.checkpoint_file = self._compute_checkpoint_path(params["checkpoint"])
            self.checkpoint_file = os.path.join(path,"models","weights",checkpoint)
            self.model.load_state_dict(torch.load(self.checkpoint_file))

    def _reshape_x_y(self, x, y):
        x = x.reshape(x.shape[0], -1, 3)
        x = x.transpose(0, 2, 1)
        y = np.expand_dims(y, axis=2)
        return x, y

    def _reshape_and_create_tensor(self, x, y):
        x, y = self._reshape_x_y(x, y)
        return TensorDataset(Tensor(x).cuda(), Tensor(y).cuda())


class FCN_Encoder_Module(nn.Module):
    def __init__(self, n_in, channels, kernel_sizes, dropout):
        super(FCN_Encoder_Module, self).__init__()
        input_dims = self._compute_input_dims(n_in, channels)
        self.encoder = _create_sequential(input_dims, channels, kernel_sizes, dropout)

    def forward(self, input):
        return self.encoder(input)

    def _compute_input_dims(self, n_in, channels):
        return [n_in] + channels[:-1]


class FCN_Regressor_Module(nn.Module):
    def __init__(self, input_dims, channels, kernel_sizes, dropout):
        super(FCN_Regressor_Module, self).__init__()
        self.regressor = _create_sequential(input_dims, channels, kernel_sizes, dropout)
        self.regressor.add_module("conv_pred_last", nn.Conv1d(channels[-1], 1, 1))

    def forward(self, features):
        return self.regressor(features)


class FCN_Domain_Classifier_Module(nn.Module):
    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    def __init__(self, input_dims, channels, kernel_sizes, dropout, n_domains):
        super(FCN_Domain_Classifier_Module, self).__init__()
        self.domain_classifier = _create_sequential(input_dims, channels, kernel_sizes, dropout)
        self.domain_classifier.add_module("conv_domain_last", nn.Conv1d(channels[-1], n_domains, 1))
        self.domain_classifier.add_module("logsoftmax", nn.LogSoftmax(dim=1))

    def forward(self, features):
        reverse_features = self.GradReverse.apply(features)
        return self.domain_classifier(reverse_features)


class DANN_FCN_Module(nn.Module):
    def __init__(self, n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout, decoder_channels,
                 decoder_dropout, n_domains):
        super(DANN_FCN_Module, self).__init__()

        decoder_input_dims = [encoder_channels[-1]]
        decoder_kernel_sizes = [self._compute_decoder_kernel_size(encoder_kernel_sizes)]

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

    def _compute_decoder_kernel_size(self, encoder_kernel_sizes, pooling=1):
        kernel_size = hist_freq
        for encoder_kernel_size in encoder_kernel_sizes:
            kernel_size -= (encoder_kernel_size - 1) + 1 * (pooling - 1) + 1
            kernel_size = np.ceil(kernel_size / pooling + 1)
        return int(kernel_size)


class DANN_Loss(nn.Module):
    def __init__(self, lambda_):
        super(DANN_Loss, self).__init__()

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
        # return mse, mse, nll



def _create_conv_layer(input_dim, channels, kernel_size, dropout):
    return [
        nn.Conv1d(input_dim, channels, kernel_size),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(channels),
        nn.Dropout(dropout)
    ]


def _create_sequential(input_dims, channels, kernel_sizes, dropout):
    return nn.Sequential(*np.concatenate(
        [_create_conv_layer(input_dim, channel, kernel_size, dropout) for input_dim, channel, kernel_size
         in zip(input_dims, channels, kernel_sizes)]))
