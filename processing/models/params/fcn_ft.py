parameters = {
    "domain_adversarial": False,
    "da_lambda": 0, #10**(-0.75),
    "domain_weights": False,
    "hist": 180,
    "n_in": 3,

    "encoder_channels": [64, 128, 64],
    "encoder_kernel_sizes": [3, 3, 3],
    "encoder_dropout": 0.5,

    "decoder_channels": [2048],
    "decoder_dropout": 0.5,

    # training_old
    "epochs": 2,
    "batch_size": 100,
    "lr": 1e-5,  # 1e-3
    "patience": 50,

    "l2": 0.0,  # 1e-2,

    "checkpoint": None,
}