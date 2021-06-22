class BaseRunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def _parse_images(self, batch):
        imgs = batch.get('image', None)
        if imgs is not None:
            imgs = imgs.to(self.device)
        return imgs
    
    def _parse_fmaps(self, batch):
        fmap_targets = batch.get('feature_map', None)
        if fmap_targets is not None:
            fmap_targets = fmap_targets.to(self.device)
            fmap_targets = fmap_targets.unsqueeze(1)
        return fmap_targets
    
    def _parse_targets(self, batch):
        targets = batch.get('target', None)
        if targets is not None:
            targets = targets.to(self.device)
            targets = targets.unsqueeze(1)
        return (targets) 

    def _parse_data(self, batch):
        raise NotImplementedError

    def _evaluate_step(self, batch):
        raise NotImplementedError
    
    def _train_step(self, batch):
        raise NotImplementedError

    def _visualize_step(self, batch):
        raise NotImplementedError

