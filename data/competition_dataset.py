from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from . import register_dataset


@register_dataset('Competition')
class CompetitionDataset(Dataset):
    def __init__(self,
                 data,
                 fmap_info,
                 mode,
                 preprocessor
                ):
        self.data = data
        # TODO: maybe move fmap instanciation to loss handler
        self.reutrn_fmap = fmap_info['return']
        if self.reutrn_fmap:
            self.fmap_size = fmap_info['size']
            if fmap_info['gauss']:
                x = np.arange(0, self.fmap_size)
                y = np.arange(0, self.fmap_size)
                xx, yy = np.meshgrid(x, y)
                x_shift = (x.max() - x.min()) / 2
                y_shift = (y.max() - y.min()) / 2
                sigma = x.max() / fmap_info['gauss_sigma']
                self.pos_feature_map = np.exp(-((xx - x_shift)**2 + (yy - y_shift)**2) / sigma**2).clip(0., 1.)
            else:
                self.pos_feature_map = np.ones((self.fmap_size, self.fmap_size))
            self.neg_feature_map = np.zeros((self.fmap_size, self.fmap_size))

            self.pos_feature_map = torch.tensor(self.pos_feature_map, dtype=torch.float)
            self.neg_feature_map = torch.tensor(self.neg_feature_map, dtype=torch.float)
        
        self.preprocessor = preprocessor
        self.mode = mode
        if self.mode == 'train':
            self._preprocess_func = self.preprocessor.train
        elif self.mode == 'test':
            self._preprocess_func = self.preprocessor.inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self._get_images_and_labels(index)
        image_name = self._get_image_names(index)
        feature_map = self._get_feature_map(label)
        item_dict = {}
        item_dict['image'] = image
        item_dict['img_name'] = image_name
        if label is not None:
            item_dict['target'] = label
        if feature_map is not None:
            item_dict['feature_map'] = feature_map
        return item_dict

    def _get_images_and_labels(self, img_idx):
        label = self.data[img_idx].get('label', None)
        image = self._read_image(img_idx)
        image = self._preprocess_func(image) 
        if label is not None:
            label = torch.tensor(label, dtype=torch.float)
        return image, label
    
    def _get_image_names(self, img_idx):
        image_name = self.data[img_idx].get('img_name', None)
        return image_name
    
    def _get_feature_map(self, label):
        if self.reutrn_fmap and label is not None:
            if label == 1:
                feature_map = self.pos_feature_map
            elif label == 0:
                feature_map = self.neg_feature_map
        else:
            feature_map = None
        return feature_map

    def _read_image(self, img_idx):
        image_path = self.data[img_idx]['img_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @classmethod
    def create_dataset(cls, mode, preprocessor, cfg, data_cfg=None, file_path=None):
        return_dev_set = False
        if mode == 'train' and data_cfg is not None:
            if (('dev_size' in data_cfg.keys()) and (data_cfg.build_dev)):
                return_dev_set = data_cfg.dev_size > 0

        data_path = cfg.data_path
        if file_path is None:
            info_file = data_cfg.data_info_file
            file_path = str(Path(data_path, info_file))
        
        data_info = []
        with open(file_path, 'r') as out_f:
            for line in out_f:
                data_info.append(line.rstrip().split(' '))

        data_by_id = defaultdict(list) if return_dev_set else None
        data = []  
        # for rel_path, label in train_info:
        
        for sample_data in data_info:
            rel_path = sample_data[0]

            if len(sample_data) == 2:
                label = int(sample_data[1])
            else:
                label = None

            if mode == 'train':
                img_id = '_'.join(Path(rel_path).parts[1].split('_')[:2])
            else:
                img_id = None
            
            img_dict = {}
            img_dict['img_name'] = rel_path
            img_dict['img_path'] = str(Path(data_path, rel_path))
            img_dict['label'] = label
            img_dict['id'] = img_id
            
            data.append(img_dict)
            
            if data_by_id is not None:
                data_by_id[img_id].append(img_dict)

        fmap_info = {
            'return':      cfg.model.fmap_out,
            'size':        cfg.backbone.fmap_size,
            'gauss':       cfg.model.fmap_out_params.gauss_fmap,
            'gauss_sigma': cfg.model.fmap_out_params.gauss_sigma
        }

        if return_dev_set:
            ids = list(data_by_id.keys())

            id_idxs = list(range(len(ids)))
            
            # train-test split by ids
            train_idxs, dev_idxs = train_test_split(id_idxs, test_size=data_cfg.dev_size)
            train_data = []
            for train_idx in train_idxs:
                train_data.extend([imd for imd in data_by_id[ids[train_idx]]])

            dev_data = []
            for dev_idx in dev_idxs:
                dev_data.extend([imd for imd in data_by_id[ids[dev_idx]]])

            train_ds = cls(train_data, fmap_info, 'train', preprocessor)
            val_ds = cls(dev_data, fmap_info, 'test', preprocessor)
            return {'train': train_ds, 'val': val_ds}
        else:
            return {mode: cls(data, fmap_info, mode, preprocessor)}