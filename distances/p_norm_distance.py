import torch
import torch.nn.functional as F

class PNormDistance:
    def __init__(self, p=2):
        self.p = p

    def calculate_distance_matrix(self, in_tensors):
        tensors_num = len(in_tensors)
        distances = torch.pow(in_tensors, self.p).sum(dim=1, keepdim=True).expand(tensors_num, tensors_num)
        distances = distances + distances.t()
        distances.addmm_(in_tensors, in_tensors.t(), beta=1, alpha=-2)
        distances = distances.clamp(min=1e-12).sqrt()
        return distances

    def calculate_distance(self, in_tensor_1, in_tensor_2):
        return F.pairwise_distance(in_tensor_1, in_tensor_2, self.p)