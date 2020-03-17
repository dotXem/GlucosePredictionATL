import torch
from misc.utils import printd
import os
from processing.models.deep_predictor import DeepPredictor
import misc.constants as cs
from .pytorch_tools.training import fit, predict
from .pytorch_tools.fcn_submodules import *


class FCN(DeepPredictor):
    def fit(self, weights_file=None, tl_mode="target_training"):
        if not tl_mode == "target_global":
            # get training_old data
            x_train, y_train, t_train = self._str2dataset("train")
            x_valid, y_valid, t_valid = self._str2dataset("valid")

            # save model
            rnd = np.random.randint(1e7)
            self.checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", "fcn_" + str(rnd) + ".pt")
            printd("Saved model's file:", self.checkpoint_file)

            if self.params["domain_adversarial"]:
                n_domains = int(np.max(y_train[:,1]) + 1)
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
                self.loss_func = self.DALoss(self.params["da_lambda"])
            else:
                self.loss_func = nn.MSELoss()

            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

            train_ds = self.to_dataset(x_train, y_train)
            valid_ds = self.to_dataset(x_valid, y_valid)

            fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds,
                valid_ds,
                self.params["patience"], self.checkpoint_file)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self.to_dataset(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        # predict
        # y_pred = self.model.predict(x, batch_size=self.params["batch_size"])
        y_true, y_pred = predict(self.model, ds)

        return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        time_step = data.time.diff().max()

        # extract data from data df
        t = data["datetime"]
        if self.params["domain_adversarial"]:
            y = data[["y","domain"]].values
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

    class FCN_Module(nn.Module):
        def __init__(self, n_in, history_length, encoder_channels, encoder_kernel_sizes, encoder_dropout,
                     decoder_channels, decoder_dropout, domain_adversarial=False, n_domains=0):
            super().__init__()

            decoder_input_dims = [encoder_channels[-1]]
            decoder_kernel_sizes = [compute_decoder_kernel_size(encoder_kernel_sizes, history_length)]

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

    class DALoss(nn.Module):
        def __init__(self, lambda_):
            super().__init__()

            self.lambda_ = lambda_

            self.mse = nn.MSELoss()
            self.nll = nn.NLLLoss()

        def forward(self, x, y):
            y_preds = x[0]
            domain_preds = x[1]

            y_trues = y[:, 0]
            domain_trues = y[:, 1].long()

            mse = self.mse(y_preds, y_trues)
            nll = self.nll(domain_preds, domain_trues)

            return mse + self.lambda_ * nll, mse, nll
