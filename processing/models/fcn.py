import numpy as np
import torch
import torch.nn as nn
import os
from processing.models.deep_tl_predictor import DeepTLPredictor
from processing.models.pytorch_tools.fcn_creation import FCN_Encoder_Module, FCN_Regressor_Module, \
    FCN_Domain_Classifier_Module, _compute_decoder_kernel_size
from .pytorch_tools.training import fit, predict


class FCN(DeepTLPredictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)

        self.model = self.FCN_Module(*self.input_shape, self.params["encoder_channels"],
                                     self.params["encoder_kernel_sizes"],
                                     self.params["encoder_dropout"], self.params["decoder_channels"],
                                     self.params["decoder_dropout"], self.params["domain_adversarial"], self.n_domains)
        self.model.cuda()

    def fit(self):
        x_train, y_train, _ = self._str2dataset("train")
        x_valid, y_valid, _ = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        self.loss_func = self._compute_loss_func()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds,
            valid_ds, self.params["patience"], self.checkpoint_file)

    def predict(self, dataset, clear=True):
        x, y, t = self._str2dataset(dataset)
        ds = self._to_tensor_ds(x, y)

        # reload the last checkpoint saved
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        if self.params["domain_adversarial"]:
            [y_trues_glucose, y_trues_subject], [y_preds_glucose, y_preds_subject] = predict(self.model, ds)
            results = self._format_results_source(y_trues_glucose, y_trues_subject, y_preds_glucose, y_preds_subject, t)
        else:
            y_true, y_pred = predict(self.model, ds)
            results = self._format_results(y_true, y_pred, t)

        if clear:
            self._clear_checkpoint()

        return results

    def save(self, file):
        no_da_fcn = self.FCN_Module(*self.input_shape, self.params["encoder_channels"],
                                    self.params["encoder_kernel_sizes"],
                                    self.params["encoder_dropout"], self.params["decoder_channels"],
                                    self.params["decoder_dropout"], False, 1)
        no_da_fcn.encoder.load_state_dict(self.model.encoder.state_dict())
        no_da_fcn.regressor.load_state_dict(self.model.regressor.state_dict())
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        torch.save(no_da_fcn.state_dict(), file)

    def _reshape(self, data):
        x, y, t = super()._reshape(data)
        x = x.transpose(0, 2, 1)  # order: batch, channels, history

        return x, y, t

    def _compute_input_shape(self):
        x_train, _, _ = self._str2dataset("train")
        return x_train.shape[1], x_train.shape[2]

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



