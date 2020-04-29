parameters = {
    "hist": 180,
    "kernel": "rbf",
    "C": [1e0, 1e3],
    "epsilon": [1e-3,1e0],
    "gamma": [1e-4, 1e-2],
    # "C": [1e-2, 1e3],
    # "epsilon": [1e-3,1e-1],
    # "gamma": [1e-5, 1e-3],
}

search = {
    "C": ["logarithmic", 4, 3],
    "epsilon": ["logarithmic", 4,3],
    "gamma": ["logarithmic", 3,3],
}
