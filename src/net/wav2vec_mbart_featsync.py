
import random
import numpy as np 

import torch

from transformers import MBartModel

class W2VMBartFS(MBartModel):
    def __init__(self, config, seed=66):
        super().__init__(config)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

