from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import yaml


def max_min_normalization(x, _max, _min):
    r"""
    Max-min normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = 1. * (x - _min)/(_max - _min)                 # [0,1]

    return x


def read_data(output_dir):

    config_path = "configs/S4.yaml"
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    feature_names = ['S4', 'ELE', 'Lat', 'Lon', 'Bt', 'MgII', 'Solar_ELE', 'Solar_AZI', 'SYMH', 'ASYH', 'Flow_Speed']

    min_max = pd.read_csv(output_dir + 'Min_Max.csv', header=None)
    min_max = min_max.values
    vmin = min_max[:, 0]
    vmax = min_max[:, 1]

    vnames = feature_names
    vmax = dict(zip(vnames, vmax))
    vmin = dict(zip(vnames, vmin))

    with open(output_dir + 'vmax.pkl', 'wb') as file:
        pickle.dump(vmax, file)
    with open(output_dir + 'vmin.pkl', 'wb') as file:
        pickle.dump(vmin, file)


    Loss_mask = pd.read_csv(output_dir + 'Mask.csv', header=None)

    for i in range(len(feature_names)):
        if config['feature_args'][feature_names[i]]:
            file_path = output_dir + feature_names[i] + '.csv'
            min_ = vmin[feature_names[i]]
            max_ = vmax[feature_names[i]]
            feature_names[i] = pd.read_csv(file_path, header=None)

            feature_names[i][Loss_mask == 0] = max_min_normalization(feature_names[i][Loss_mask == 0], max_, min_)
            feature_names[i] = np.around(feature_names[i].values, 4)
            feature_names[i] = np.expand_dims(feature_names[i], axis=-1)

            if i == 0:
                feature_list = [feature_names[i]]
            else:
                feature_list.append(feature_names[i])


    data = np.concatenate(feature_list, axis=-1)

    return data


def generate_graph_seq2seq_io_data(
        data, Idx, output_dir
):
    """
    Generate samples from
    """

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    # Input
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = [], []
    min_t = abs(min(x_offsets))

    for n in range(Idx.shape[0]):
        md = data[(Idx[n, 0] - 1).item():Idx[n, 1], ...]
        ser_num_samples = md.shape[0]
        max_t = abs(ser_num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x.append(md[t + x_offsets, ...])
            y.append(md[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)


    return x, y



def generate_train_val_test(args):

    output_dir = args.output_dir

    data = read_data(output_dir);

    Idx = pd.read_csv(output_dir + 'Idx.csv', header=None)              # random day index
    Idx = Idx.values
    day_num = Idx.shape[0]          # 1900

    num_train = round(day_num * 0.7)
    num_val = round(day_num * 0.8)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x_train, y_train = generate_graph_seq2seq_io_data(data = data, Idx = Idx[0:num_train], output_dir = output_dir)
    x_val, y_val =  generate_graph_seq2seq_io_data(data = data, Idx = Idx[num_train:num_val], output_dir = output_dir)
    x_test, y_test = generate_graph_seq2seq_io_data(data = data, Idx = Idx[num_val:], output_dir = output_dir)


    # Write the data into npz file.
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y
        )


if __name__ == "__main__":

    seq_length_x    = 36
    seq_length_y    = 12
    y_start         = 1
    dataset         = "S4"
    output_dir  = 'datasets/S4/'
    Data_filename = 'S4'
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory.")
    parser.add_argument("--Data_filename", type=str, default=Data_filename, help="S4.",)
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=y_start, help="Y pred start", )

    
    args    = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply   = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
