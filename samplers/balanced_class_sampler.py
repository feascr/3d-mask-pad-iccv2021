import torch
from torch.utils.data import Sampler
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np

class BalancedClassSampler(Sampler):
    def __init__(self, dataset, batch_size, batches_per_iteration, replacement=True, generator=None):
        self.num_samples = len(dataset)
        self.batch_size = batch_size
        self.batches_per_iteration = batches_per_iteration
        
        self.weights = self._calculate_weights(dataset)
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.batch_size * self.batches_per_iteration, replacement=self.replacement, generator=self.generator)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.batch_size * self.batches_per_iteration

    @staticmethod
    def _calculate_weights(dataset):
        data = dataset.data
        labels = [img_dict['label'] for img_dict in data]
        
        _, counts = np.unique(labels, return_counts=True)
        counts = counts.astype(float)
        
        weights = sum(counts) / counts
        weights = [weights[label] for label in labels]
        return torch.tensor(weights)