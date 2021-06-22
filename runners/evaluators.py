from tqdm import tqdm
import torch
from enum import Enum
from itertools import combinations, chain
from .base_runner import BaseRunner


class FuseFunction(Enum):
    MIN = 'min'
    MEAN = 'mean'
    NONE = 'none'


class BaseEvaluator(BaseRunner):
    def __init__(self, model, loss_handler, metric_calculator, device, fuse_function=FuseFunction.NONE):
        super(BaseEvaluator, self).__init__(model, device)
        self.loss_handler = loss_handler
        self.fuse_function_name = fuse_function.value 
        if fuse_function == FuseFunction.NONE:
            self._fuse_function = None
        elif fuse_function == FuseFunction.MIN:
            self._fuse_function = self._fuse_function_min
        elif fuse_function == FuseFunction.MEAN:
            self._fuse_function = self._fuse_function_mean
        self.metric_calculator = metric_calculator

    def get_loss_state_dicts(self):
        return self.loss_handler.get_state_dicts()

    def evaluate(self, dev_iterators_dict):
        self.model.eval()
        if self.loss_handler is not None:
            iterators_loss_dict = {}
        else:
            iterators_loss_dict = None

        iterators_metrics_dict = {}
        
        with torch.no_grad():
            for iterator_key, dev_iterator in dev_iterators_dict.items():
                if iterators_loss_dict is not None:
                    iterators_loss_dict[iterator_key] = {}
                iterator_outputs_dict = {}
                iterator_targets_list = []

                for batch in tqdm(dev_iterator):
                    loss_dict, outputs_dict, targets = self._evaluate_step(batch)
                    
                    for output_key in outputs_dict:
                        if output_key not in iterator_outputs_dict.keys():
                            iterator_outputs_dict[output_key] = []
                        iterator_outputs_dict[output_key].append(outputs_dict[output_key])
                    
                    iterator_targets_list.append(targets)

                    if iterators_loss_dict is not None:
                        for loss_key in loss_dict:
                            if loss_dict[loss_key] is not None:
                                if loss_key not in iterators_loss_dict[iterator_key].keys():
                                    iterators_loss_dict[iterator_key][loss_key] = 0.
                                iterators_loss_dict[iterator_key][loss_key] += loss_dict[loss_key].detach().cpu().numpy() * dev_iterator.batch_size
                
                if iterators_loss_dict is not None:
                    for loss_key in iterators_loss_dict[iterator_key]:
                        iterators_loss_dict[iterator_key][loss_key] = iterators_loss_dict[iterator_key][loss_key] / len(dev_iterator.dataset)

                iterator_targets_list = torch.cat(iterator_targets_list).cpu().numpy()

                iterators_metrics_dict[iterator_key] = {}
                for output_key in iterator_outputs_dict:
                    iterator_outputs_dict[output_key] = torch.cat(iterator_outputs_dict[output_key]).cpu().numpy()
                    iterators_metrics_dict[iterator_key][output_key] = self.metric_calculator.calculate_metrcis(
                        iterator_outputs_dict[output_key], 
                        iterator_targets_list
                    )
        return iterators_loss_dict, iterators_metrics_dict

    @staticmethod
    def _fuse_function_min(predictions_list):
        predictions = torch.cat(predictions_list, dim=1)
        predictions = torch.min(predictions, dim=1, keepdim=True)[0]
        return predictions

    @staticmethod
    def _fuse_function_mean(predictions_list):
        predictions = torch.cat(predictions_list, dim=1)
        predictions = torch.mean(predictions, dim=1, keepdim=True)
        return predictions

    def _parse_data(self, batch):
        imgs = self._parse_images(batch)
        fmap_targets = self._parse_fmaps(batch)
        targets = self._parse_targets(batch)
        return (imgs, fmap_targets, targets)

    def _evaluate_step(self, batch):
        raise NotImplementedError

class BasicEvaluator(BaseEvaluator):
    def _evaluate_step(self, batch):
        outputs_dict = {}
        # Load data to selected device
        imgs, fmaps_targets, targets = self._parse_data(batch)
        
        # Calculate predictions
        model_out_dict = self.model(imgs)
        # Calculate loss
        if self.loss_handler is not None:
            loss_dict = self.loss_handler.calculate_loss(**model_out_dict, fmaps_targets=fmaps_targets, targets=targets)
        else:
            loss_dict = None

        if 'fmaps' in model_out_dict:
            fmap_preds = model_out_dict['fmaps'].mean(dim=[2, 3])
            fmap_preds = fmap_preds.sigmoid()
            # fmap_preds = model_out_dict['fmaps'].sigmoid()
            # fmap_preds = fmap_preds.mean(dim=[2, 3])
            outputs_dict['fmaps'] = fmap_preds
        if 'logits' in model_out_dict:
            logit_preds = model_out_dict['logits'].sigmoid()
            outputs_dict['logits'] = logit_preds
        
        if self.loss_handler is not None:
            outputs_dict.update(self.loss_handler.get_loss_predictions(model_out_dict))
        
        if self._fuse_function is not None:
            out_keys = list(outputs_dict.keys())
            if len(out_keys) > 1:
                fuse_keys_iter = chain.from_iterable(combinations(out_keys, r) for r in range(2, len(out_keys) + 1))
                for fuse_keys in fuse_keys_iter:
                    predicts_to_fuse = []
                    for out_key in fuse_keys:
                        predicts_to_fuse.append(outputs_dict[out_key])
                    outputs_dict['{}_fused_{}'.format('_'.join(fuse_keys), self.fuse_function_name)] = self._fuse_function(predicts_to_fuse)
                

        return loss_dict, outputs_dict, targets