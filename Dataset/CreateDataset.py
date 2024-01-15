import numpy as np
import pickle as pkl
import pandas as pd
import os 
import xarray
import matplotlib.pyplot as plt
from yaml import load, Loader
import sys

config_path = sys.argv[1]

def filter_dates(ki_maps, maps):
    ki_time_idx = ki_maps.time.values
    maps_time_idx = maps.time.values
    t_idx = [i for i in range(len(maps_time_idx)) if maps_time_idx[i] in set(ki_time_idx)]
    return maps[t_idx]

def filter_dates(ki_maps, maps):
    ki_time_idx = ki_maps.time.values
    maps_time_idx = maps.time.values
    t_idx = [i for i in range(len(maps_time_idx)) if maps_time_idx[i] in set(ki_time_idx)]
    return maps[t_idx]

def get_coord_idx(img_shape, patch_size):
    x_shape = img_shape[1]
    y_shape = img_shape[0]
    
    x_slide = (x_shape - (x_shape//patch_size)*patch_size)//2
    y_slide = (y_shape - (y_shape//patch_size)*patch_size)//2
    
    starting_x_idx = [c for c in np.arange(x_slide, x_shape, patch_size) if c+patch_size<x_shape]
    starting_y_idx = [c for c in np.arange(y_slide, y_shape, patch_size) if c+patch_size<y_shape]
    starting_idx = np.array([[(y,x) for x in starting_x_idx] for y in starting_y_idx]).reshape(-1,2)
    return starting_idx




def main(config_path):
    with open(config_path, 'rb') as o:
        config = load(o, Loader)
    
    ki_data_path = config['ki_data_path']
    sza_data_path = config['sza_data_path']

    train_years = config['train_years']
    val_years = config['val_years']
    test_years = config['test_years']
    seq_len = config['seq_len']
    training_set_save_path = config['training_set_save_path']
    validation_set_save_path = config['validation_set_save_path']
    test_set_save_path = config['test_set_save_path']

    # define starting coordinates
    starting_idx = get_coord_idx(img_shape=(396, 804), patch_size=128)

    # get all dates
    dates = sorted([f.split('_')[1].split('.')[0] for f in os.listdir(ki_data_path)])

    for date in dates:
        print(date)
        ki_file_path = ki_data_path + 'ki_{}.pkl'.format(date)
        sza_file_path = sza_data_path + 'sza_{}.pkl'.format(date)
        year = date[:4]

    if year in train_years:
        save_path = training_set_save_path
    elif year in val_years:
        save_path = validation_set_save_path
    elif year in test_years:
        save_path = test_set_save_path

    with open(ki_file_path, 'rb') as o:
        ki_file = pkl.load(o)

    with open(sza_file_path, 'rb') as o:
        sza_file = pkl.load(o)

    # GET STATS ON KI MAPS
    ki_maps = ki_file['maps']
    nan_idx = ki_file['nan_idx']
    ki_nan_maps = np.array(ki_maps).copy()
    ki_nan_maps[nan_idx] = np.nan

    sza_maps = np.array(filter_dates(ki_maps, sza_file['maps']))

    ki_maps_sequences = np.array([ki_maps[:, x[0]:x[0]+128, x[1]:x[1]+128] for x in starting_idx])
    ki_nan_maps_sequences = np.array([ki_nan_maps[:, x[0]:x[0]+128, x[1]:x[1]+128] for x in starting_idx])
    sza_maps_sequences = np.array([sza_maps[:, x[0]:x[0]+128, x[1]:x[1]+128] for x in starting_idx])
    # print(ki_maps_sequences.shape, ki_nan_maps_sequences.shape, sza_maps_sequences.shape)

    time_sequences = np.array([ki_maps.time.values for x in starting_idx])
    time_sequences = np.array(pd.to_datetime(time_sequences).round('min'))

    # ITERATION ON THE SEQUENCES
    n_patches = time_sequences.shape[0]
    full_seq_len = time_sequences.shape[1]
    for i in range(n_patches):
        # retrieve possible starting points
        starting_idx_lst = []
        for j in range(full_seq_len-seq_len):
            # retrieve time relative to the sequence
            t = time_sequences[i, j:j+seq_len]
            ki_nan_patch_seq = ki_nan_maps_sequences[i, j:j+seq_len]
            nan_check_seq = np.array([np.isnan(ki_nan_patch_seq[ii]).sum()/(128**2) for ii in range(seq_len)])
            # retrieve sequence length
            minutes = int((t[-1] - t[0])/6e+10)

            if minutes == (seq_len-1)*15 and all(nan_check_seq<0.25):
                starting_idx_lst.append(j)
        
        ki_maps_seq = ki_maps_sequences[i]
        print(len(ki_maps_seq), (len(starting_idx_lst)+seq_len)/len(ki_maps_seq))
        sza_seq = np.nanmean(sza_maps_sequences[i].reshape(full_seq_len,-1), axis=(1))

        # save dict with the maps and spatiotemporal coordinates
        ki_sample_dict = {'ki_maps':ki_maps_seq,
                          'time':time_sequences[i],
                          'sza':sza_seq,
                          'coord_idx':j,
                          'starting_idx':starting_idx_lst}

        with open(save_path+'KI/{}_{}.pkl'.format(date, i), 'wb') as o:
             pkl.dump(ki_sample_dict, o)

if __name__ == '__main__':
    main(config_path)