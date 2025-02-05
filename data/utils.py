import torch
import numpy as np


# https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
def worker_init_func(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))