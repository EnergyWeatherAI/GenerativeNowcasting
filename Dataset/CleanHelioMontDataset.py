import xarray
import pickle as pkl
import os
from yaml import load, Loader
import numpy as np
from scipy import interpolate
import time
from multiprocessing import Pool
from functools import partial
import sys 

config_path = sys.argv[1]


def fillna(arr, method='nearest'):
    t = np.arange(0, arr.shape[0])
    y = np.arange(0, arr.shape[1])
    x = np.arange(0, arr.shape[2])
    array = np.ma.masked_invalid(arr)

    tt, yy, xx = np.meshgrid(y, t, x)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    t1 = tt[~array.mask]
    newarr = array[~array.mask]

    interp_arr = interpolate.griddata((t1, y1, x1),
                                      newarr.ravel(),
                                      (tt, yy, xx),
                                      method=method)

    arr.data = interp_arr


def clean_and_save_file(maps_data, fill_method, day, dir_path):
    var_name, maps = maps_data
    nanIndex = None
    if np.isnan(maps).any():
        # print('filling {} NaN ...'.format(var_name))
        nanIndex = np.where(np.isnan(maps))
        fillna(maps, method=fill_method)
    path = dir_path + '{}/'.format(var_name.upper()) + '{}_'.format(var_name.lower()) + day + '.pkl'
    with open(path, 'wb') as o:
        pkl.dump({'maps': maps, 'nan_idx': nanIndex}, o)


def open_and_clean(nc_file, day, dir_path, thresh=0.25, fill_method='nearest', save_latlon=False):
    ki_maps = nc_file.KI
    sis_maps = nc_file.SIS
    siscf_maps = nc_file.SISCF
    cm_maps = nc_file.CMASK
    cmerr_maps = nc_file.CMASKERROR
    sza_maps = nc_file.SZA

    acceptable_maps_idx = []
    for i, m in enumerate(ki_maps):
        if np.sum(np.isnan(m)) / np.prod(m.shape) < thresh:
            acceptable_maps_idx.append(i)

    ki_maps = ki_maps[acceptable_maps_idx]
    sis_maps = sis_maps[acceptable_maps_idx]
    siscf_maps = siscf_maps[acceptable_maps_idx]

    config_list = [('KI', ki_maps), ('SIS', sis_maps), ('SISCF', siscf_maps),
                   ('CM', cm_maps), ('CMERR', cmerr_maps), ('SZA', sza_maps)]

    partial_clean_and_save_file = partial(clean_and_save_file, fill_method=fill_method,
                                          day=day, dir_path=dir_path)
    print('process started')
    with Pool() as pool:
        pool.map(partial_clean_and_save_file, config_list)

    if save_latlon:
        lat = nc_file.lat
        lon = nc_file.lon
        with open(dir_path + 'lat_lon.pkl', 'wb') as o:
            pkl.dump({'lat': lat, 'lon': lon}, o)
        print('latlon saved')


def main(config_path):
    with open(config_path, 'rb') as o:
        config = load(o, Loader)
    
    dir_path = config['save_path']
    data_path = config['data_path']

    filename_lst = sorted(os.listdir(data_path))
    for i, filename in enumerate(filename_lst):
        try:
            nc_file = xarray.open_dataset(data_path + filename).copy()
        except:
            print(filename, 'corrupted data')
        day = filename.split('.')[-2]
        if not np.isnan(nc_file.KI.data).all():
            print(day)
            if i == 0:
                save_latlon = True
            else:
                save_latlon = False

            open_and_clean(nc_file, day, dir_path, thresh=0.25, fill_method='nearest', save_latlon=save_latlon)
        else:
            print(day, 'all NaNs')

if __name__ == '__main__':
    main(config_path)