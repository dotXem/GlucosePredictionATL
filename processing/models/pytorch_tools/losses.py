import torch.nn as nn


class DALoss(nn.Module):
    def __init__(self, lambda_, weight=None):
        super().__init__()

        self.lambda_ = lambda_
        self.weight = weight

        self.mse = nn.MSELoss()
        self.nll = nn.NLLLoss(weight=weight)

    def forward(self, x, y):
        y_preds = x[0]
        domain_preds = x[1]

        y_trues = y[:, 0]
        domain_trues = y[:, 1].long()

        mse = self.mse(y_preds, y_trues)
        nll = self.nll(domain_preds, domain_trues)

        return mse + self.lambda_ * nll, mse, nll