import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from misc.utils import printd
import os
from processing.models.deep_predictor import DeepPredictor
import misc.constants as cs
from processing.models.pytorch_tools.gradient_reversal import RevGrad
from .pytorch_tools.training import fit, predict
from .pytorch_tools.losses import DALoss


class FCN(DeepPredictor):
    def fit(self, weights_file=None, tl_mode="target_training", save_file=None):
        # get training_old data
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")

        # save model
        rnd = np.random.randint(1e7)
        self.checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", "fcn_" + str(rnd) + ".pt")
        printd("Saved model's file:", self.checkpoint_file)

        if self.params["domain_adversarial"]:
            n_domains = int(np.max(y_train[:, 1]) + 1)
            if self.params["domain_weights"]:
                domains_weights = Tensor([np.sum(y_train[:, 1] == i) / len(y_train)
                                          for i in range(int(max(y_train[:, 1])) + 1)])
                domains_weights = 1 / domains_weights
                domains_weights /= domains_weights.min()
                domains_weights = domains_weights.cuda()
            else:
                domains_weights = None
        else:
            n_domains = 1

        self.model = self.FCN_Module(x_train.shape[1], x_train.shape[2], self.params["encoder_channels"],
                                     self.params["encoder_kernel_sizes"],
                                     self.params["encoder_dropout"], self.params["decoder_channels"],
                                     self.params["decoder_dropout"], self.params["domain_adversarial"], n_domains)
        self.model.cuda()

        if weights_file is not None:
            self.load_weights_from_file(weights_file)

        if self.params["domain_adversarial"]:
            self.loss_func = DALoss(self.params["da_lambda"], domains_weights)
        else:
            self.loss_func = nn.MSELoss()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        train_ds = self.to_dataset(x_train, y_train)
        valid_ds = self.to_dataset(x_valid, y_valid)

        if not tl_mode == "target_global":
            fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds,
                valid_ds,
                self.params["patience"], self.checkpoint_file)

            if tl_mode == "source_training" and save_file is not None:
                self.save_fcn(save_file)

    def predict(self, dataset, clear=True):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self.to_dataset(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        if self.params["domain_adversarial"]:
            [y_trues_glucose, y_trues_subject], [y_preds_glucose, y_preds_subject] = predict(self.model, ds)
            y_trues_glucose, y_preds_glucose = [_.reshape(-1, 1) for _ in [y_trues_glucose, y_preds_glucose]]
            y_trues_subject = y_trues_subject.reshape(-1, 1)
            y_preds_subject = np.argmax(y_preds_subject, axis=1).reshape(-1, 1)
            y_true, y_pred = np.c_[y_trues_glucose, y_trues_subject], np.c_[y_preds_glucose, y_preds_subject]

            if clear:
                self.clear_checkpoint()

            return self._format_results_source(y_true, y_pred, t)
        else:
            y_true, y_pred = predict(self.model, ds)

            if clear:
                self.clear_checkpoint()

            return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        time_step = data.time.diff().max()

        # extract data from data df
        t = data["datetime"]
        if self.params["domain_adversarial"]:
            y = data[["y", "domain"]].values
        else:
            y = data["y"].values

        g = data.loc[:, [col for col in data.columns if "glucose" in col]].values
        cho = data.loc[:, [col for col in data.columns if "CHO" in col]].values
        ins = data.loc[:, [col for col in data.columns if "insulin" in col]].values

        time_step_arr = np.flip(np.arange(g.shape[1]) * time_step).reshape(1, -1)
        t_x = data["time"].values.reshape(-1, 1) * np.ones((1, np.shape(time_step_arr)[1])) - time_step_arr

        # reshape timeseties in (n_samples, hist, 1) shape and concatenate
        g = g.reshape(-1, g.shape[1], 1)
        cho = cho.reshape(-1, g.shape[1], 1)
        ins = ins.reshape(-1, g.shape[1], 1)
        # t_x = t_x.reshape(-1, g.shape[1], 1)

        # x = np.concatenate([g, cho, ins, t_x], axis=2)
        x = np.concatenate([g, cho, ins], axis=2)
        x = x.transpose(0, 2, 1)  # order: batch, channels, history

        return x, y, t

    def save_fcn(self, save_file):
        x_train, y_train, t_train = self._str2dataset("train")
        no_da_fcn = self.FCN_Module(x_train.shape[1], x_train.shape[2], self.params["encoder_channels"],
                                    self.params["encoder_kernel_sizes"],
                                    self.params["encoder_dropout"], self.params["decoder_channels"],
                                    self.params["decoder_dropout"], False, 1)
        no_da_fcn.encoder.load_state_dict(self.model.encoder.state_dict())
        no_da_fcn.regressor.load_state_dict(self.model.regressor.state_dict())
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        torch.save(no_da_fcn.state_dict(), save_file)

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)

        self.model = self.FCN_Module(x.shape[1], x.shape[2], self.params["encoder_channels"],
                                     self.params["encoder_kernel_sizes"],
                                     self.params["encoder_dropout"], self.params["decoder_channels"],
                                     self.params["decoder_dropout"], False, 0)
        self.model.cuda()
        self.model.load_state_dict(torch.load(file))

        self.model.eval()
        features = self.model.encoder(Tensor(x).cuda()).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]

    class FCN_Module(nn.Module):
        def __init__(self, n_in, history_length, encoder_channels, encoder_kernel_sizes, encoder_dropout,
                     decoder_channels, decoder_dropout, domain_adversarial=False, n_domains=1):
            super().__init__()

            decoder_input_dims = [encoder_channels[-1]]
            decoder_kernel_sizes = [_compute_decoder_kernel_size(encoder_kernel_sizes, history_length)]

            self.encoder = FCN_Encoder_Module(n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout)
            self.regressor = FCN_Regressor_Module(decoder_input_dims, decoder_channels, decoder_kernel_sizes,
                                                  decoder_dropout)

            if domain_adversarial:
                self.domain_classifier = FCN_Domain_Classifier_Module(decoder_input_dims, decoder_channels,
                                                                      decoder_kernel_sizes, decoder_dropout, n_domains)
            else:
                self.domain_classifier = None

        def forward(self, input):
            features = self.encoder(input)
            prediction = self.regressor(features)
            if self.domain_classifier is not None:
                domain = self.domain_classifier(features)
                return prediction.squeeze(), domain.squeeze()
            else:
                return prediction.squeeze()


""" TOOLS FOR BUILDING THE FCN_MODULE """


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
    def __init__(self, input_dims, channels, kernel_sizes, dropout, n_domains):
        super(FCN_Domain_Classifier_Module, self).__init__()
        self.domain_classifier = nn.Sequential(
            RevGrad(),
            *np.concatenate(
                [_create_conv_layer(input_dim, channel, kernel_size, dropout) for input_dim, channel, kernel_size
                 in zip(input_dims, channels, kernel_sizes)]),
            nn.Conv1d(channels[-1], n_domains, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, features):
        return self.domain_classifier(features)


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


def _compute_decoder_kernel_size(encoder_kernel_sizes, history_length, pooling=1):
    kernel_size = history_length
    for encoder_kernel_size in encoder_kernel_sizes:
        kernel_size -= (encoder_kernel_size - 1) + 1 * (pooling - 1) + 1
        kernel_size = np.ceil(kernel_size / pooling + 1)
    return int(kernel_size)
