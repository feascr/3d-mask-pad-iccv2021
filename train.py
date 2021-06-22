import time, argparse, subprocess
from pathlib import Path
from shutil import copyfile
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import yaml
from addict import Dict
import logging

from data import create_dataset
from data.utils import worker_init_func
from preprocessors import create_preprocessor
from metrics import MetricCalculator
from auxiliary import seed_everything, iteration_time, configure_logger, configure_logger_file_handler
from models import create_model
import loss
import runners
import samplers
import lr_schedulers


def fit_model(trainer, evaluator, train_iterator, dev_iterators_dict, scheduler, num_iterations, save_path):
    # Initialize best metric score
    best_valid_metrics = {}

    train_losses = {}
    valid_losses = {}
    valid_metric_scores = {}

    for set_key in dev_iterators_dict:
        valid_losses[set_key] = {}
        valid_metric_scores[set_key] = {}
        best_valid_metrics[set_key] = {}
    # Main loop
    for iteration in range(1, num_iterations + 1):
        start_time = time.time()

        # Step of scheduler        
        scheduler.step()

        training_lr = optimizer.param_groups[0]["lr"]

        logging.info('Iteration: {:d} | Training:'.format(iteration))
        # Training loop
        train_loss_dict = trainer.train(train_iterator)
        logging.info('Iteration: {:d} | Validating:'.format(iteration))
        # Validation loop
        dev_loss_dict, dev_metrics_dict = evaluator.evaluate(dev_iterators_dict)

        # Step of scheduler
        # scheduler.step()

        for loss_key in train_loss_dict:
            if loss_key not in train_losses:
                train_losses[loss_key] = []
            train_losses[loss_key].append(train_loss_dict[loss_key])

        for set_key in dev_loss_dict:
            for loss_key in dev_loss_dict[set_key]:
                if loss_key not in valid_losses[set_key]:
                    valid_losses[set_key][loss_key] = []
                valid_losses[set_key][loss_key].append(dev_loss_dict[set_key][loss_key])
        
        for set_key in dev_metrics_dict:
            if set_key not in valid_metric_scores:
                valid_metric_scores[set_key] = {}
            for out_key in dev_metrics_dict[set_key]:
                if out_key not in valid_metric_scores[set_key]:
                    valid_metric_scores[set_key][out_key] = {}
                for metric_key in dev_metrics_dict[set_key][out_key]['target']:
                    if metric_key not in valid_metric_scores[set_key][out_key]:
                        valid_metric_scores[set_key][out_key][metric_key] = []
                    valid_metric_scores[set_key][out_key][metric_key].append(
                        dev_metrics_dict[set_key][out_key]['target'][metric_key]
                    )

        # Saving model with best metric score
        save_model = False
        set_metric_names = []
        save_metric_scores = []
        for set_key in dev_metrics_dict:
            if set_key not in best_valid_metrics:
                best_valid_metrics[set_key] = {}
            # if 'train_dev' in set_key:
            for metric_key in dev_metrics_dict[set_key]:
                if metric_key not in best_valid_metrics[set_key]:
                    best_valid_metrics[set_key][metric_key] = {'EER': np.Inf, 'ACER': np.Inf}
                if dev_metrics_dict[set_key][metric_key]['target']['EER'] <= best_valid_metrics[set_key][metric_key]['EER']:
                    save_model = True
                    set_metric_names.append((set_key, metric_key, 'EER'))
                    save_metric_scores.append(
                        {
                            'old': best_valid_metrics[set_key][metric_key]['EER'], 
                            'new': dev_metrics_dict[set_key][metric_key]['target']['EER']
                        }
                    )
                    best_valid_metrics[set_key][metric_key]['EER'] = dev_metrics_dict[set_key][metric_key]['target']['EER']
                if dev_metrics_dict[set_key][metric_key]['target']['ACER'] <= best_valid_metrics[set_key][metric_key]['ACER']:
                    save_model = True
                    set_metric_names.append((set_key, metric_key, 'ACER'))
                    save_metric_scores.append(
                        {
                            'old': best_valid_metrics[set_key][metric_key]['ACER'], 
                            'new': dev_metrics_dict[set_key][metric_key]['target']['ACER']
                        }
                    )
                    best_valid_metrics[set_key][metric_key]['ACER'] = dev_metrics_dict[set_key][metric_key]['target']['ACER']
            
        if save_model and save_path is not None:
            logging.info('\n') #DEBUG
            for set_metric_tuple, metric_score in zip(set_metric_names, save_metric_scores):
                logging.info(
                    '{} {} on set {} decreased ({:.6f} --> {:.6f}).'.format(
                        set_metric_tuple[1],
                        set_metric_tuple[2], 
                        set_metric_tuple[0], 
                        metric_score['old'], 
                        metric_score['new']
                    )
                )
            logging.info('Saving model ...')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.get_optimizer_state_dict(),
                'loss_state_dicts': evaluator.get_loss_state_dicts()
                }, '{}/iteration_{:d}.pt'.format(save_path, iteration)
            )

        end_time = time.time()
        iteration_mins, iteration_secs = iteration_time(start_time, end_time)

        # Iteration information
        content = '\nIteration: {:d}, lr: {:.7f} | Iteration Time: {:d}m {:d}s\n'.format(iteration, 
                                                                                         training_lr, 
                                                                                         iteration_mins, 
                                                                                         iteration_secs)
        for loss_key in train_loss_dict:
            content += '\tTrain {}: {:.5f}'.format(loss_key, train_loss_dict[loss_key]) 
        
        content += '\n'
        all_sets_keys = set(list(dev_loss_dict.keys()) + list(dev_metrics_dict.keys()))
        for set_key in all_sets_keys:
            content += '\n\t{}:\n'.format(set_key)
            if set_key in dev_loss_dict:
                for loss_key in dev_loss_dict[set_key]:
                    content += '\tTest.  {}: {:.5f}'.format(loss_key, 
                                                            dev_loss_dict[set_key][loss_key])
                content += '\n'

            if set_key in dev_metrics_dict:
                for out_key in dev_metrics_dict[set_key]:
                    content += '\tMetrics for {} predictions\n\t'.format(out_key)
                    for metric_key in dev_metrics_dict[set_key][out_key]['target']:
                        content += '\tTest. {}: {:.5f}'.format(metric_key, 
                                                               dev_metrics_dict[set_key][out_key]['target'][metric_key])
                    content += '\n'

        content += '\n'

        logging.info(content)
        
    return (train_losses, valid_losses, valid_metric_scores)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', type=str, default='./configs/train_config.yml', help='path to config file')
    parser.add_argument('--name-suffix', type=str, help='additional suffix to results folder name')
    parser.add_argument('--device', type=str, default=None, help='device name')
    parser.add_argument('--nf-crop', action='store_true')
    args = parser.parse_args()

    # config file opening
    with open(args.config) as cfg_f:
        # load dict keys as class attributes
        config = Dict(yaml.safe_load(cfg_f))

    # seed everything
    seed_everything(config.seed)

    configure_logger()
    # set device
    device_name = config.device if args.device is None else args.device
    device = torch.device(device_name if (torch.cuda.is_available() and 'cuda' in device_name) else 'cpu')
    logging.debug('Initialized device: {}'.format(device))

    preprocessor = create_preprocessor(
        config.preprocessor_name, 
        config
    )
    
    # build model
    model = create_model(config.model.name, config)
    model = model.to(device)
    if hasattr(model, 'get_fmap_size'):
        config.backbone.fmap_size = model.get_fmap_size()
    else:
        config.backbone.fmap_size = None

    # build data_sets
    train_ds_dict, dev_ds_dict = {}, {}
    ds_list = [train_ds_dict, dev_ds_dict]
    mode_list = ['train', 'test']
    cfg_data_list = [config.train_data, config.dev_data]
    for ds_dict, mode, cfg_data in zip(ds_list, mode_list, cfg_data_list):
        for data_element_cfg in cfg_data:
            ds_data = create_dataset(data_element_cfg.dataset_name, mode, preprocessor, config, data_element_cfg)
            if mode == 'train':
                ds_dict[data_element_cfg['meta_name']] = ds_data['train']
                if 'val' in ds_data.keys():
                    dev_ds_dict[data_element_cfg['meta_name'] + '_val'] = ds_data['val']
            else:
                ds_dict[data_element_cfg['meta_name']] = ds_data[mode]

    logging.debug('\nAll data sets are loaded:')
    logging.debug('\tTrain data sets:')
    for k, v in train_ds_dict.items():
        logging.debug('\t{} with {} images'.format(k, len(v)))
    logging.debug('\tTest data sets:')
    for k, v in dev_ds_dict.items():
        logging.debug('\t{} with {} images'.format(k, len(v)))
    

    # build dataloaders
    if 'cpu' not in '{}'.format(device):
        pin_memory = True
    else:
        pin_memory = False
    # Using only first train set !
    train_ds = train_ds_dict[config.train_data[0].meta_name]
    # build simple weight sampler based on class distribution0
    sampler = samplers.BalancedClassSampler(
        train_ds, 
        config.batch_size, 
        config.batches_per_iteration
    )
        
    train_iterator = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=pin_memory, 
        persistent_workers=False,
        worker_init_fn=worker_init_func
    )
    
    dev_iterators_dict = {}
    for k, v in dev_ds_dict.items():
        dev_iterators_dict[k] = DataLoader(
            v, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            drop_last=False, 
            pin_memory=pin_memory, 
            persistent_workers=False,
            worker_init_fn=worker_init_func
        )

    logging.debug('\nDataLoaders are built with pin_memory: {}'.format(pin_memory))

    # build metric calculator
    metric_calculator = MetricCalculator() 
    logging.debug('\nMetric Calculator is built')

    logging.debug('\nModel {} is built and loaded'.format(config.model.name))

    if config.model.pretrained_path is not None:
        state_dict = torch.load(config.model.pretrained_path, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        logging.debug('\nModel {} pretrained weights are loaded'.format(config.model.name))
    
    # build loss
    loss = loss.LossHandler(config, device) 
    logging.debug('\nLoss is built and loaded')

    # build optmizer
    optim_parameters = [{'params': model.parameters()}] 
    if hasattr(model, 'get_additional_named_parameters'):
        for name, parameters in model.get_additional_named_parameters():
            lr = config.optim.get('{}_lr'.format(name), config.optim.lr)
            optim_parameters += [{'params': parameters, 'lr': lr}]    
    optim_parameters += loss.get_optimized_loss_parameters()
    if config.optim.name == 'Adam':
        optimizer = optim.Adam(optim_parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    else:
        raise ValueError('{} is not implemented'.format(config.optim.name))
    
    # build scheduler
    if config.scheduler.get('cosine_restart_mult', None) is not None:
        scheduler = lr_schedulers.CosineAnnealingAnnealedWarmRestarts(
            restart_mult=config.scheduler.cosine_restart_mult,
            optimizer=optimizer, 
            T_0=config.scheduler.cosine_t_0, 
            T_mult=config.scheduler.cosine_t_mult
        )
    elif config.scheduler.get('cosine_t_0', None) is not None and config.scheduler.get('cosine_t_mult', None) is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.scheduler.cosine_t_0, 
            T_mult=config.scheduler.cosine_t_mult
        )
    else:
        scheduler = scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            config.num_iterations - config.scheduler.warmup_iters + 1
        )
    scheduler_warmup = lr_schedulers.GradualWarmupScheduler(
        optimizer, 
        multiplier=config.scheduler.warmup_multiplier,
        total_epoch=config.scheduler.warmup_iters, 
        after_scheduler=scheduler
    )
    logging.debug('\nOptimizer {} is built and loaded'.format(config.optim.name))

    trainer = runners.BasicTrainer(
        aug_type=config.online_augmentations.type,
        alpha=config.online_augmentations.alpha,
        aug_prob=config.online_augmentations.prob,
        model=model,
        optimizer=optimizer,
        loss_handler=loss,
        device=device
    )

    evaluator = runners.BasicEvaluator(
        model, 
        loss, 
        metric_calculator, 
        device, 
        fuse_function=runners.FuseFunction(config.fuse_function.lower())
    )

    # prepare folder for saving results
    if args.name_suffix:
        suffix_name = '_' + args.name_suffix
    else:
        suffix_name = ''
    save_path = config.save_path
    if save_path is not None:
        res_folder_name = Path(save_path, 'results', (str(int(time.time())) + '_' + config.model.name + suffix_name))
        res_folder_name.mkdir(exist_ok=True)
        
        train_config_path = res_folder_name / 'train_config.yml'
        _ = copyfile(args.config, train_config_path)
        
        configure_logger_file_handler(str(Path(res_folder_name, 'training_log.txt')))
        logging.debug('\nResults dir is made: {}\n'.format(res_folder_name))

        git_commit_hash_path = Path(res_folder_name, 'git_commit_hash.txt')
        with open(git_commit_hash_path, 'w') as out_file:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
            out_file.write('{}\n'.format(commit_hash))

        res_folder_name = str(res_folder_name)
    else:
        res_folder_name = None
        logging.debug('\nResults will not be saved\n')
    
    # start main train loop
    train_losses, valid_losses, valid_metric_scores = fit_model(
        trainer, 
        evaluator, 
        train_iterator, 
        dev_iterators_dict,
        scheduler_warmup, 
        config.num_iterations, 
        res_folder_name
    )