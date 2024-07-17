from tqdm import tqdm 
import torch
import random
import numpy as np
import torch.nn as nn
from torch_geometric.nn import Node2Vec

def set_random(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)


def label_smoothing(labels, epsilon, num_classes):
    soft_labels = torch.full((labels.size(0), num_classes), fill_value=epsilon / (num_classes - 1), device=labels.device)
    soft_labels.scatter_(1, labels.unsqueeze(1), (1 - epsilon) + epsilon / (num_classes - 1))
    
    return soft_labels


class EarlyStopping:
    def __init__(self, path, patience=10, min_delta=0):
        """
        :param patience: Can tolerate no improvement within how many epochs
        :param min_delta: The minimum amount of change for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf') # the smaller the better
        self.early_stop = False
        self.path = path

    def __call__(self, model_to_save, score):
        if score >= self.best_score - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            torch.save(model_to_save, self.path)


def clone_module(module, memo=None):
    """

    Source: https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

    """

    if memo is None:
        memo = {}

    # First, create a copy of the module.
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                module._parameters[param_key] = updated

    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key], updates=None, memo=memo)

    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module
