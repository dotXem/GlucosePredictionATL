import os
import torch
from torch.utils.data import TensorDataset
from torch import Tensor
from training.pytorch.training import fit, predict
from models.tools.fcn_submodules import *
from models.tools.files import compute_checkpoint_file_name, compute_checkpoint_path, compute_weights_path

params = {
    "n_in": 3,

    "encoder_channels": [64, 128, 64],
    "encoder_kernel_sizes": [3, 3, 3],
    "encoder_dropout": 0.9,

    "decoder_channels": [2048],
    "decoder_dropout": 0.9,

    # training
    "epochs": 5000,
    "bs": 100,
    "lr": 1e-4,
    "patience": 50,

    "l2": 0.0,

    "checkpoint": None,
}


class FCN():
    def __init__(self, params):
        self.params = params

        self.model = FCN_Module(n_in=self.params["n_in"],
                                encoder_channels=self.params["encoder_channels"],
                                encoder_kernel_sizes=self.params["encoder_kernel_sizes"],
                                encoder_dropout=self.params["encoder_dropout"],
                                decoder_channels=self.params["decoder_channels"],
                                decoder_dropout=self.params["decoder_dropout"])

        self.model.cuda()

        self._create_and_load_checkpoint(self.params["checkpoint"])

        self.loss = nn.MSELoss()

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

    def predict(self, x, y):
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        ds = self._reshape_and_create_tensor(x, y)

        y_trues, y_preds = predict(self.model, ds)

        return np.array(y_trues).reshape(-1, 1), np.array(y_preds).reshape(-1, 1)

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

    def load_weights(self, encoder_weights, regressor_weights):
        self.model.encoder.load_state_dict(encoder_weights)
        self.model.regressor.load_state_dict(regressor_weights)
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def save_weights(self, file_name):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        torch.save(self.model.state_dict(), compute_weights_path(file_name))

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


class FCN_Module(nn.Module):
    def __init__(self, n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout, decoder_channels,
                 decoder_dropout):
        super(FCN_Module, self).__init__()

        decoder_input_dims = [encoder_channels[-1]]
        decoder_kernel_sizes = [compute_decoder_kernel_size(encoder_kernel_sizes)]

        self.encoder = FCN_Encoder_Module(n_in, encoder_channels, encoder_kernel_sizes, encoder_dropout)
        self.regressor = FCN_Regressor_Module(decoder_input_dims, decoder_channels, decoder_kernel_sizes,
                                              decoder_dropout)

    def forward(self, input):
        features = self.encoder(input)
        prediction = self.regressor(features)

        return prediction