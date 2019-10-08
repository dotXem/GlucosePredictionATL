import torch
import numpy as np
from torch.utils.data import DataLoader
from tools.pytorch.early_stopping import EarlyStopping
from tools.printd import printd


def loss_batch(model, loss_func, xb, yb, opt=None):
    # loss = loss_func(model(xb), yb)
    loss, mse, nll = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), mse.item(), nll.item(), len(xb)
    # return loss.item(), len(xb)


def eval_loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb)[0], yb[:, 0].unsqueeze(dim=1))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, batch_size, model, loss_func, opt, train_ds, valid_ds, patience, checkpoint_file):
    train_dl, valid_dl = create_dataloaders_from_datasets(train_ds, valid_ds, batch_size)

    early_stopping = EarlyStopping(patience=patience,
                                   path=checkpoint_file)

    model.eval()
    early_stopping, res = evaluate(0, early_stopping, model, loss_func, [train_dl, valid_dl])

    for epoch in range(epochs):
        model.train()
        zip(*[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl])

        model.eval()
        early_stopping, res = evaluate(epoch, early_stopping, model, loss_func, [train_dl, valid_dl])

        if early_stopping.early_stop:
            printd("Early Stopped.")
            break

    early_stopping.save()


def evaluate(epoch, early_stopping, model, loss_func, dls):
    dls_names = ["[train]", "[valid]"]
    with torch.no_grad():
        loss = []
        for dl, name in zip(dls, dls_names):
            import torch.nn as nn
            # losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
            # loss_dl = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            # loss.append(loss_dl)
            #
            # if name == "[valid]":
            #     early_stopping(loss_dl, model, epoch)

            losses, mse, nll, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])

            sum = np.sum(nums)
            loss_dl = [np.sum(np.multiply(losses, nums)) / sum, np.sum(np.multiply(mse, nums)) / sum,
                       np.sum(np.multiply(nll, nums)) / sum]
            loss.append(loss_dl)

            if name == "[valid]":
                early_stopping(loss_dl[1], model, epoch)

    res = np.r_[epoch, np.c_[dls_names, loss].ravel()]
    printd(*res)

    return early_stopping, res


def predict(model, ds):
    # TODO check rework with predict in DANN_FCN.py
    model.eval()
    dl = DataLoader(ds, batch_size=len(ds))

    # trues = dl.dataset.tensors[1].cpu().numpy()
    # trues = cuda2np([dl.dataset.tensors[1]])
    # preds = model(dl.dataset.tensors[0]).cpu().detach().numpy()
    preds = cuda2np(model(dl.dataset.tensors[0]))


    trues = np.rollaxis(dl.dataset.tensors[1].cpu().detach().numpy(),1,0)
    preds = [ts.cpu().detach().numpy() for ts in model(dl.dataset.tensors[0])]

    return trues, preds


def cuda2np(tensors):
    np_tensors = []
    for tensor in tensors:
        np_tensors.append(tensor.cpu().detach().numpy())
    return np_tensors


def create_dataloaders_from_datasets(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
