import argparse
import numpy as np
from tqdm import tqdm
import yaml
from addict import Dict
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models import create_model
from data import create_dataset
from data.utils import worker_init_func
from preprocessors import create_preprocessor


def evaluate(iterators_by_file, model, predictions_from, device, save_path):
    # iterations loop
    results = {'img_names': [], 'img_proba': []}
    assert list(iterators_by_file.keys())[0] == 'val.txt' or 'val.txt' not in iterators_by_file.keys()
    for file_key, iterator in iterators_by_file.items():
        with torch.no_grad():
            for batch in tqdm(iterator):
                imgs = batch['image']
                img_names = batch['img_name']

                imgs = imgs.to(device)
                out_dict = model(imgs)
                if predictions_from == 'fmaps':
                    fmaps = out_dict.get('fmaps', None)
                    if fmaps is not None:
                        res = fmaps.mean(dim=[2, 3])
                        probability = res.sigmoid()
                    else:
                        raise ValueError('Feature maps is not in models_out_dict') 
                elif predictions_from == 'logits':
                    logits = out_dict.get('logits', None)
                    if logits is not None:
                        res = logits
                        probability = res.sigmoid()
                    else:
                        raise ValueError('Logits is not in models_out_dict')
                probability = probability.detach().cpu().numpy()
                if probability.ndim == 2:
                    probability = probability.squeeze(1).tolist()
                results['img_names'].extend(img_names)
                results['img_proba'].extend(probability)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / '{}.txt'.format('_'.join(iterators_by_file.keys())), 'w') as out_f:
        for img_name, img_proba in zip(results['img_names'], results['img_proba']):
            out_f.write('{} {}\n'.format(img_name.split('/')[-1], img_proba))
                 
def main(config: Dict):
    train_config = Dict(config['train_config'])
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    iteration = checkpoint['iteration']
    model = create_model(train_config.model.name, train_config)
    model = model.to(config['device'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    if hasattr(model, 'get_fmap_size'):
        train_config['backbone']['fmap_size'] = model.get_fmap_size()
    else:
        train_config['backbone']['fmap_size'] = None
    
    preprocessor = create_preprocessor(train_config.preprocessor_name, train_config)
    data_path = config['data_path']
    info_files = config['info_files']
    iterators_by_file = {}
    for info_file in info_files:
        dataset = create_dataset(
            'Competition', 
            'test', 
            preprocessor, 
            train_config, 
            file_path=str(Path(data_path, info_file))
        )['test']
        iterators_by_file[str(Path(info_file).stem)] = DataLoader(
            dataset,
            drop_last=False,
            batch_size=config['train_config'].batch_size,
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=False,
            worker_init_fn=worker_init_func
        )
        
    save_path = Path(config['checkpoint_path']).parent / 'it_{}'.format(iteration) / config['predictions_from']
    evaluate(iterators_by_file, model, config['predictions_from'], config['device'], save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test submitter')
    parser.add_argument('-c', '--config', type=str, default='./configs/evaluate_config.yml', help='Path to configuration file (JAML)')
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_f:
        config = yaml.safe_load(cfg_f)

    result_directory = Path(config['result_directory'])
    for i, experiment_info in enumerate(config['experiments']):
        experiment_dir = Path(experiment_info['checkpoint_path']).parent
        experiment_dir = result_directory / experiment_dir

        for f_path in experiment_dir.glob('*.yml'):
            train_config_path = f_path
            with open(f_path, 'r') as cfg_f:
                # load dict keys as class attributes
                train_config = Dict(yaml.safe_load(cfg_f))
        config['train_config_path'] = str(train_config_path)
        config['train_config'] = train_config
        config['checkpoint_path'] = str(Path(result_directory, experiment_info['checkpoint_path']))
        config['predictions_from'] = experiment_info['predictions_from']
        
        print('Processing experiment: {}'.format(i))
        main(config)