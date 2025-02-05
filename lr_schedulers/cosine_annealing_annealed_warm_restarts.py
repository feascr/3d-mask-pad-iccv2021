import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

class CosineAnnealingAnnealedWarmRestarts(CosineAnnealingWarmRestarts):
    """Identical to PyTorch CosineAnnealingWarmRestarts 
    implementation except this scheduler decrease max lr
    by `restart_mult` for each warm restart.
    """
    def __init__(self, restart_mult, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        self.restart_mult = restart_mult
        self.num_rest = 0
        super(CosineAnnealingAnnealedWarmRestarts, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.T_cur == 0 and self.last_epoch > 0:
            # increase restart count
            self.num_rest += 1
        return [self.eta_min + (base_lr - self.eta_min) * (self.restart_mult ** self.num_rest) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2 
                for base_lr in self.base_lrs]