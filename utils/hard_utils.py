import random
import numpy as np
import torch
import os


def is_key_in_dict(d, key):
    try:
        d[key]
    except KeyError:
        return False
    return True


def set_seed(seed=0xD153A53):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
