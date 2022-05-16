import os

import torch
import pytorch_lightning


def seed_everything(seed=42):
    pytorch_lightning.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
