parameters = {
    "domain_adversarial": False,
    "domain_weights": False,
    "da_lambda": 0,
    "hist": 180,

    # model hyperparameters
    "hidden": [256,256],

    # training_old hyperparameters
    "dropout_weights":0.,
    "dropout_layer":0.0,
    "epochs": 2,
    "batch_size": 50,
    "lr": 1e-5,
    "l2": 0.0, #1e-4,
    "patience": 50,
}
