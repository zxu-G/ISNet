#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pickle
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def re_normalization(x, mean, std):
    r"""
    Standard re-normalization

    mean: float
        Mean of data
    std: float
        Standard of data
    """
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    r"""
    Max-min normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = 1. * (x - _min)/(_max - _min)
    # x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    r"""
    Max-min re-normalization

    _max: float
        Max
    _min: float
        Min
    """
    # x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class StandardScaler():
    r"""
    Description:
    -----------
    Standard the input.

    Args:
    -----------
    mean: float
        Mean of data.
    std: float
        Standard of data.

    Attributes:
    -----------
    Same as Args.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    r"""
    Description:
    -----------
    Load pickle data.
    
    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data




class Dataset_init(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        # self.loss_mask = torch.Tensor(loss_mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _   = np.load(os.path.join(data_dir, mode + '.npz'))
        data_dict['x_' + mode]  = _['x']
        data_dict['y_' + mode]  = _['y']
        # data_dict['loss_mask_' + mode] = _['loss_mask']

    worker_num = 10
    train_dataset = Dataset_init(data_dict['x_train'], data_dict['y_train'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    data_dict['train_loader']   = DataLoader(train_dataset, batch_size, num_workers=worker_num, pin_memory=True, sampler=train_sampler)
    data_dict['val_loader']     = DataLoader(Dataset_init(data_dict['x_val'], data_dict['y_val']), valid_batch_size, shuffle=False, num_workers=worker_num, pin_memory=True)
    data_dict['test_loader']    = DataLoader(Dataset_init(data_dict['x_test'], data_dict['y_test']), test_batch_size, shuffle=False, num_workers=worker_num, pin_memory=True)
    data_dict['scaler']         = re_max_min_normalization

    return data_dict


