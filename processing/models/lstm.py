from torch import Tensor
from .pytorch_tools.lstm_variants import LSTM_Gal
from .pytorch_tools.gradient_reversal import RevGrad
from torch.utils.data import TensorDataset
import torch
from misc.utils import printd
import os
import re
from processing.models.deep_predictor import DeepPredictor
import numpy as np
import misc.constants as cs
import torch.nn as nn
from .pytorch_tools.training import fit, predict
from .pytorch_tools.losses import DALoss

#TODO clean after results

class LSTM(DeepPredictor):
    def fit(self, weights_file=None, tl_mode="target_training", save_file=None):
        # get training_old data
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")

        # save model
        rnd = np.random.randint(1e7)
        self.checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", "lstm_" + str(rnd) + ".pt")
        printd("Saved model's file:", self.checkpoint_file)

        if self.params["domain_adversarial"]:
            n_domains = int(np.max(y_train[:, 1]) + 1)
        else:
            n_domains = 1

        self.model = self.LSTM_Module(x_train.shape[2], self.params["hidden"], self.params["dropi"],
                                      self.params["dropw"], self.params["dropo"], self.params["domain_adversarial"], n_domains, self.params["da_lambda"])
        self.model.cuda()

        if weights_file is not None:
            self.load_weights_from_file(weights_file)

        if self.params["domain_adversarial"]:
            self.loss_func = DALoss(self.params["da_lambda"])
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

        return x, y, t

    def save_fcn(self, save_file):
        x_train, y_train, t_train = self._str2dataset("train")
        no_da_fcn = self.LSTM_Module(x_train.shape[2], self.params["hidden"], self.params["dropi"],
                                          self.params["dropw"], self.params["dropo"], False)
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        no_da_fcn.encoder.load_state_dict(self.model.encoder.state_dict())
        no_da_fcn.regressor.load_state_dict(self.model.regressor.state_dict())
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        torch.save(no_da_fcn.state_dict(), save_file)

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)

        self.model = self.LSTM_Module(x.shape[2], self.params["hidden"], self.params["dropi"],
                                          self.params["dropw"], self.params["dropo"], False)
        # self.model.cuda()
        self.model.load_state_dict(torch.load(file))

        self.model.eval()
        # features = self.model.encoder(Tensor(x).cuda())[0].detach().cpu().numpy()
        features = self.model.encoder(Tensor(x))[0].detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]

    class LSTM_Module(nn.Module):

        def __init__(self, n_in, neurons, dropi, dropw, dropo, domain_adversarial=False, n_domains=1, lambda_=0):
            super().__init__()
            # TODO handle embedding dropout and layer dropout for LSTM_Gal

            # if embedding_size is not None:
            #     self.emb_linear = nn.Linear(n_in, embedding_size, bias=False)
            #     self.emb_drop = nn.Dropout(embedding_dropout)
            # else:
            #     self.emb_linear = None
            #     embedding_size = n_in

            # self.lstm = BETTER_LSTM(n_in, neurons[0], len(neurons), dropouti=dropi, dropoutw=dropw, dropouto=dropo, batch_first=True)

            # if not dropw == 0.0:
            #     # self.lstm = [LSTM_Gal(embedding_size, neurons[0], recurrent_dropout, batch_first=True)]
            #     self.lstm = [LSTM_Gal(n_in, neurons[0], dropw, batch_first=True).cuda()]
            #     self.dropouts = []
            #     for i in range(len(neurons[1:])):
            #         self.dropouts.append(nn.Dropout(dropo))
            #         self.lstm.append(LSTM_Gal(neurons[i], neurons[i + 1], dropw, batch_first=True).cuda())
            #         # if (not i == len(neurons[1:]) - 1):
            #     self.dropouts.append(nn.Dropout(0.0))
            #     self.lstm = nn.Sequential(*self.lstm)
            #     self.dropouts = nn.Sequential(*self.dropouts)
            # else:
                # self.lstm = nn.LSTM(embedding_size, neurons[0], len(neurons), dropout=layer_dropout, batch_first=True)
                # self.lstm = nn.LSTM(n_in, neurons[0], len(neurons), dropout=dropo, batch_first=True)

            self.encoder = nn.LSTM(n_in, neurons[0], len(neurons), dropout=dropo, batch_first=True)

            self.regressor = nn.Linear(neurons[-1], 1)

            self.domain_classifier = nn.Sequential(
                RevGrad(lambda_),
                nn.Linear(neurons[-1], n_domains),
                nn.LogSoftmax(dim=1),
            ) if domain_adversarial else None

            # print(self.lstm)
            # r = 3
            # q = 512
            # p = q
            # m = 3
            # dp = 0.95

            # batch_first=True => input/ouput w/ shape (batch,seq,feature)
            # self.lstm = cs.LSTM(r,q)
            # self.lstm = LSTM_Gani(r, q, dp, True)
            # self.lstm = LSTM_Semenuita(r, q, dp, True)
            # self.lstm = nn.LSTM(r,q,1, batch_first=True)

            # self.linear = nn.Linear(q, 1)
            # self.dropout = nn.Dropout(0.25)

        # from torch.autograd import Function
        # class GradReverse(Function):
        #     @staticmethod
        #     def forward(ctx, x):
        #         return x.view_as(x)
        #
        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         return grad_output.neg()

        def forward(self, xb):
            # xb, _ = self.lstm(xb)
            # xb = self.dropout(xb[:, -1, :])
            # xb = self.linear(xb)

            # if self.emb_linear is not None:
            #     xb = self.emb_drop(self.emb_linear(xb))

            # if self.lstm.__class__.__name__ == "LSTM":
            #     xb, _ = self.lstm(xb)
            # else:
            #     for lstm_, dropout_ in zip(self.lstm, self.dropouts):
            #         xb = lstm_(xb)[0]
            # xb = self.linear(xb[:, -1, :])
            # return xb.reshape(-1)

            features, _ = self.encoder(xb)
            prediction = self.regressor(features[:,-1])
            if self.domain_classifier is not None:
                # domain = self.domain_classifier(self.GradReverse.apply(features[:,-1]))
                domain = self.domain_classifier(features[:,-1])
                return prediction.squeeze(), domain.squeeze()
            else:
                return prediction.squeeze()


    def to_dataset(self, x, y):
        # return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())
        return TensorDataset(torch.Tensor(x), torch.Tensor(y))
