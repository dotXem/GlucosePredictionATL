import os
from _misc import path
import numpy as np
from datetime import datetime

def compute_checkpoint_path(file_name):
    return os.path.join(path, "tmp", "checkpoints", file_name)

def compute_weights_path(file_name):
    return os.path.join(path, "models", "weights", file_name)

def compute_checkpoint_file_name(model):
    return datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + model.__class__.__name__ + "_" + str(
                np.random.randint(10000)) + ".pt"