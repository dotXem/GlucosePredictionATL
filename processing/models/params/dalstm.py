parameters = {
    "domain_adversarial": True,
    "da_lambda": 10**(-0.75),
    "domain_weights": True,
    "hist": 180,

    # model hyperparameters
    "hidden": [256,256],

    # training_old hyperparameters
    "dropout_weights":0.,
    "dropout_layer":0.0,
    "epochs": 2,
    "batch_size": 50,
    "lr": 1e-4,
    "l2": 0.0,
    "patience": 10,
}
