import torch
from torch import nn
from torch import optim
from .smoothed_bce_with_logits_loss import SmoothedBCEWithLogitsLoss
from distances import PNormDistance


class LossHandler:
    def __init__(self, cfg, device):
        self.losses = {}

        if cfg.model.fmap_out:
            if cfg.fmap_loss:
                self.losses['fmaps'] = self._build_losses_dicts(cfg.fmap_loss, device)
        
        if cfg.model.class_logit_out:
            if cfg.logit_loss:
                self.losses['logits'] = self._build_losses_dicts(cfg.logit_loss, device)
        
        if cfg.model.embedding_out:
            if cfg.embedding_loss:
                self._embedding_size = cfg.model.embedding_params.embedding_size
                self.losses['embeddings'] = self._build_losses_dicts(cfg.embedding_loss, device)
    
    def calculate_loss(self, **kwargs):
        calculated_loss_dict = {}
        final_loss = None
        for model_out, out_losses_dict in self.losses.items():
            try:
                data = kwargs[model_out]
            except:
                continue
            targets = kwargs['targets']
            if '{}_targets'.format(model_out) in kwargs:
                out_targets = kwargs[model_out + '_targets']
            else:
                out_targets = targets
            out_targets = (out_targets,)
            for out_loss_dict in out_losses_dict:
                if out_loss_dict['sampler'] is not None:
                    try:
                        processed_data = out_loss_dict['sampler'].sample(data, targets)
                    except:
                        calculated_loss_dict['{}_{}_loss'.format(model_out, out_loss_dict['type'])] = torch.tensor(0.)
                        continue
                else:
                    processed_data = data

                if not isinstance(processed_data, tuple):
                    processed_data = (processed_data, )
                    
                loss_args = []
                if 'processed_data' in out_loss_dict['loss_args']:
                    loss_args.extend(processed_data)
                if 'out_targets' in out_loss_dict['loss_args']:
                    loss_args.extend(out_targets)
                calculated_loss = out_loss_dict['loss'](*loss_args)
                calculated_loss_dict['{}_{}_loss'.format(model_out, out_loss_dict['type'])] = calculated_loss
                if final_loss is None:
                    final_loss = calculated_loss * out_loss_dict['coef']
                else:
                    final_loss += calculated_loss * out_loss_dict['coef']

        calculated_loss_dict['final_loss'] = final_loss
        return calculated_loss_dict

    def get_loss_predictions(self, model_out_dict):
        outputs_dict = {}
        for out_key in self.losses:
            if out_key in model_out_dict:
                for out_loss_dict in self.losses[out_key]:
                    if hasattr(out_loss_dict['loss'], 'predict'):
                        predictions = out_loss_dict['loss'].predict(model_out_dict[out_key])[1]
                        outputs_dict['{}_{}'.format(out_key, out_loss_dict['type'])] = predictions
        return outputs_dict

    def get_state_dicts(self):
        losses_state_dicts = {}
        for out_key in self.losses:
            for out_loss_dict in self.losses[out_key]:
                if out_loss_dict['optimizer_params'] is not None:
                    loss_state_dict = {}
                    if out_loss_dict['optimizer_params']['optimizer'] is not None:
                        loss_state_dict['optimizer'] =  out_loss_dict['optimizer_params']['optimizer'].__class__.__name__
                        loss_state_dict['optimizer_state_dict'] =  out_loss_dict['optimizer_params']['optimizer'].state_dict()
                    loss_state_dict['loss_state_dict'] = out_loss_dict['loss'].state_dict()
                    losses_state_dicts['{}_{}'.format(out_key, out_loss_dict['type'])] = loss_state_dict
        return losses_state_dicts

    def get_optimized_loss_parameters(self):
        parameters_list = []
        for out_key in self.losses:
            for out_loss_dict in self.losses[out_key]:
                if out_loss_dict['optimizer_params'] is not None:
                    if out_loss_dict['optimizer_params']['optimizer'] is None:
                        optim_params = {'params': out_loss_dict['loss'].parameters()}
                        if out_loss_dict['optimizer_params']['lr'] is not None:
                            optim_params.update({'lr': out_loss_dict['optimizer_params']['lr'] / out_loss_dict['coef']})
                        parameters_list.append(optim_params)
        return parameters_list

    def optimizer_zero_grad(self):
        for out_key in self.losses:
            for out_loss_dict in self.losses[out_key]:
                if out_loss_dict['optimizer_params']:
                    if out_loss_dict['optimizer_params']['optimizer'] is not None:
                        for param in out_loss_dict['loss'].parameters():
                            param.grad = None

    def optimizer_step(self):
        for out_key in self.losses:
            for out_loss_dict in self.losses[out_key]:
                if out_loss_dict['optimizer_params']:
                    if out_loss_dict['optimizer_params']['optimizer'] is not None:
                        out_loss_dict['optimizer_params']['optimizer'].step()

    def _build_losses_dicts(self, losses_cfg, device, **kwargs):
        losses_dicts = []
        for loss_cfg in losses_cfg:
            if loss_cfg.use:
                loss_type = loss_cfg.type
                loss_coef = loss_cfg.coef
                
                if 'distance' in loss_cfg:
                    distance = self._build_distance(loss_cfg.distance, **loss_cfg.distance_params)
                else:
                    distance = None
                
                loss_sampler = None
                loss, loss_args = self._build_loss(loss_cfg, distance, device)
                
                if hasattr(loss, 'need_optimization'):
                    need_optimization = loss.need_optimization
                else:
                    need_optimization = False
                if need_optimization:
                    assert 'optimizer_params' in loss_cfg, 'Must provide "optimizer_params" in config for loss parameters optimization'
                    optimizer_params = {}
                    if loss_cfg.optimizer_params.build_optimizer:
                        # build optimizer
                        optimizer_params['lr'] = loss_cfg.optimizer_params.lr
                        optimizer = optim.SGD(loss.parameters(), lr=(loss_cfg.optimizer_params.lr / loss_coef))
                        optimizer_params['optimizer'] = optimizer
                    else:
                        optimizer_params['optimizer'] = None
                        optimizer_params['lr'] = loss_cfg.optimizer_params.lr
                else:
                    optimizer_params = None

                losses_dicts.append({'type': loss_type, 'coef': loss_coef, 'sampler': loss_sampler,
                                     'loss': loss, 'loss_args': loss_args, 
                                     'optimizer_params': optimizer_params})
        return losses_dicts

    @staticmethod
    def _build_loss(loss_cfg, distance, device):
        if loss_cfg.type.lower() == 'bce':
            loss_args = ['processed_data', 'out_targets']
            if loss_cfg.label_smoothing_value:
                return SmoothedBCEWithLogitsLoss(loss_cfg.label_smoothing_value, device), loss_args
            else:
                return nn.BCEWithLogitsLoss().to(device), loss_args
        elif loss_cfg.type.lower() == 'mse':
            loss_args = ['processed_data', 'out_targets']
            return nn.MSELoss().to(device), loss_args
        else:
            raise ValueError('Loss {} is not implemented'.format(loss_cfg.type))
    
    @staticmethod
    def _build_distance(distance_type, **kwargs):
        if distance_type.lower() == 'p_norm':
            if 'p' in kwargs:
                distance = PNormDistance(kwargs['p'])
            else:
                raise ValueError('Must define parameter p for {} distance'.format(distance_type))
            return distance
        else:
            raise ValueError('{} is not implemented'.format(distance_type))