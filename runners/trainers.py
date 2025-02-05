from tqdm import tqdm
import torch
import numpy as np
import random
from .base_runner import BaseRunner


class BaseTrainer(BaseRunner):
    def __init__(self, model, optimizer, loss_handler, device):
        super(BaseTrainer, self).__init__(model, device)
        self.loss_handler = loss_handler
        self.optimizer = optimizer

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def train(self, train_iterator):
        self.model.train()
        epoch_loss_dict = {}
        for batch in tqdm(train_iterator):
            loss_dict = self._train_step(batch)

            for loss_key in loss_dict:
                if loss_dict[loss_key] is not None:
                    if loss_key not in epoch_loss_dict.keys():
                        epoch_loss_dict[loss_key] = 0.
                    epoch_loss_dict[loss_key] += loss_dict[loss_key].detach().cpu().numpy()
        
        for loss_key in epoch_loss_dict:
            epoch_loss_dict[loss_key] = epoch_loss_dict[loss_key] / len(train_iterator)
        
        return epoch_loss_dict

    def _parse_data(self, batch):
        imgs = self._parse_images(batch)
        fmap_targets = self._parse_fmaps(batch)
        targets = self._parse_targets(batch)
        return (imgs, fmap_targets, targets)

    def _train_step(self, batch):
        raise NotImplementedError


class CutMixUpTrainer(BaseTrainer):
    def __init__(self, aug_type=None, alpha=1, aug_prob=0.5, *args, **kwargs):
        if alpha < 0:
            raise ValueError('Alpha for CutMix must be greater than 0') 
        self.alpha = alpha
        self.aug_prob = aug_prob
        if aug_type is None:
            self._aug_func = self._pass_func
        elif aug_type == 'cutmix':
            self._aug_func = self._cutmix
        elif aug_type == 'mixup':
            self._aug_func = self._mixup
        else:
            raise ValueError("{} doesn't support {} augmentation".format(self.__class__.__name__, aug_type))
        super(CutMixUpTrainer, self).__init__(*args, **kwargs)

    def _cutmix(self, images, targets, fmap_targets):
        prob = random.random()
        if prob < self.cutmix_prob:
            lambda_area = np.random.beta(self.alpha, self.alpha)
            
            idxs = torch.randperm(images.shape[0])

            height = images.shape[2]
            width = images.shape[3]

            cut_rat = np.sqrt(1. - lambda_area)
            cut_w = (width * cut_rat).astype(int)
            cut_h = (height * cut_rat).astype(int)

            top_left_x = np.random.randint(0, height - cut_h + 1)
            top_left_y = np.random.randint(0, width - cut_w + 1)

            # bounding box to cut for all images
            bbx1 = np.clip(top_left_x, 0, width)
            bby1 = np.clip(top_left_y, 0, height)
            bbx2 = np.clip(top_left_x + cut_w, 0, width)
            bby2 = np.clip(top_left_y + cut_h, 0, height)

            # insert cutted regions
            images[:, :, bbx1:bbx2, bby1:bby2] = images[idxs, :, bbx1:bbx2, bby1:bby2]
            # recalculate targets
            targets = lambda_area * targets + (1 - lambda_area) * targets[idxs]
            # recalculate fmap_targets
            if fmap_targets is not None:
                fmap_targets = lambda_area * fmap_targets + (1 - lambda_area) * fmap_targets[idxs]
            return images, targets, fmap_targets
        else:
            return images, targets, fmap_targets
    
    def _mixup(self, images, targets, fmap_targets):
        prob = random.random()
        if prob < self.mixup_prob:
            lambda_mix = np.random.beta(self.alpha, self.alpha)
            
            # select indexes of image to cut from for each image
            idxs = torch.randperm(images.shape[0])
        
            # mixup images
            images = lambda_mix * images + (1 - lambda_mix) * images[idxs]
            # recalculate targets
            targets = lambda_mix * targets + (1 - lambda_mix) * targets[idxs]
            # recalculate fmap_targets
            if fmap_targets is not None:
                fmap_targets = lambda_mix * fmap_targets + (1 - lambda_mix) * fmap_targets[idxs]
            return images, targets, fmap_targets
        else:
            return images, targets, fmap_targets
    
    def _pass_func(self, images, targets, fmap_targets):
        return images, targets, fmap_targets


class BasicTrainer(CutMixUpTrainer):
    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)

    def _train_step(self, batch):
        # Load data to selected device
        imgs, fmaps_targets, targets = self._parse_data(batch)
        # Set accumulated gradients to zero
        self.loss_handler.optimizer_zero_grad()
        for param in self.model.parameters():
            param.grad = None
        # self.optimizer.zero_grad()
        imgs, targets, fmaps_targets = self._aug_func(imgs, targets, fmaps_targets)
        # Calculate predictions
        model_out_dict = self.model(imgs)
        # Calculate loss
        loss_dict = self.loss_handler.calculate_loss(
            **model_out_dict,
            targets=targets, 
            fmaps_targets=fmaps_targets
        )
        # Computed gradients for parameters
        loss_dict['final_loss'].backward()
        # Loss parameters optimization block
        # self.loss_handler.rescale_loss_parameters_grad()
        self.loss_handler.optimizer_step()
        # Step of optimizer
        self.optimizer.step()
        
        return loss_dict

