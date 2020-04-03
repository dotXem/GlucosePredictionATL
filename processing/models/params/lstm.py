parameters = {
    "domain_adversarial": False,
    "da_lambda": 0,
    "hist": 60,

    # model hyperparameters
    "hidden": [2048,2048],

    # training_old hyperparameters
    # "dropout": 0.0,
    # "recurrent_dropout": 0.3, #0.9, #0.5,
    "dropi":0.0,
    "dropw":0.0,
    "dropo":0.0,
    "epochs": 5000,
    "batch_size": 50,
    "lr": 1e-4,
    "l2": 0.0, #1e-4,
    "patience": 50,
}
