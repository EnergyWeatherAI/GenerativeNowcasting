import pickle as pkl
import numpy as np
import yaml
import torch
import torch.nn as nn

ROOT_PATH = '/Users/cea3/Desktop/Projects/GenerativeModels/'

def open_pkl(path: str):
    with open(path, 'rb') as o:
        pkl_file = pkl.load(o)
    return pkl_file


def save_pkl(path: str, obj):
    with open(path, 'wb') as o:
        pkl.dump(obj, o)


def open_yaml(path: str):
    with open(path) as o:
        yaml_file = yaml.load(o, Loader=yaml.FullLoader)
    return yaml_file


def activation(act_type="swish"):
    act_dict = {"swish": nn.SiLU(),
                "gelu": nn.GELU(),
                "relu": nn.ReLU(),
                "tanh": nn.Tanh()}
    if act_type:
        if act_type in act_dict:
            return act_dict[act_type]
        else:
            raise NotImplementedError(act_type)
    elif not act_type:
        return nn.Identity()


def normalization(channels, norm_type="group", num_groups=32):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif (not norm_type) or (norm_type.lower() == 'none'):
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)


def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def sample_from_standard_normal(mean, log_var, num=None):
    std = (0.5 * log_var).exp()
    shape = mean.shape
    if num is not None:
        # expand channel 1 to create several samples
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:, None, ...]
        std = std[:, None, ...]
    return mean + std * torch.randn(shape, device=mean.device)


# def get_full_images(date, val_data_path, coordinate_data_path, n_patches=18):

#     patches = []
#     times = []
#     coords = []
#     starting_idx_lst = []
#     for i in range(n_patches):
#         patch = open_pkl(val_data_path+date+'_'+str(i)+'.pkl')
#         lat = open_pkl(coordinate_data_path+str(i)+'_lat.pkl')
#         lon = open_pkl(coordinate_data_path+str(i)+'_lon.pkl')
#         maps = 2 * ((patch['ki_maps'] - 0.05) / (1.2 - 0.05)) - 1
#         patches.append(maps)
#         t = 2 * ((patch['sza'] - 0) / (90 - 0)) - 1
#         times.append(t)
#         lon = 2 * ((lon - 0) / (90 - 0)) - 1
#         lat = 2 * ((lat - 0) / (90 - 0)) - 1
#         coords.append((lon, lat))
#         starting_idx_lst.append(patch['starting_idx'])
#     common_starting_idx_lst = list(set.intersection(*map(set, starting_idx_lst)))
#     patches = np.array(patches)
#     patches = patches[:, 4:]
#     times = np.array(times)
#     times = np.nanmean(times[:, 4:], axis=0)

#     full_image = np.empty((patches.shape[1], 128*3, 128*6))
#     full_lat = np.empty((128*3, 128*6))
#     full_lon = np.empty((128*3, 128*6))

#     k = 0
#     for i in range(3):
#         for j in range(6):
#             full_image[:, 128*i:128*(i+1), 128*j:128*(j+1)] = patches[k]
#             full_lat[128*i:128*(i+1), 128*j:128*(j+1)] = coords[k][1]
#             full_lon[128*i:128*(i+1), 128*j:128*(j+1)] = coords[k][0]
#             k += 1
#     return full_image, full_lat, full_lon, times, common_starting_idx_lst

patch_dict = {0: ((0, 128), (0, 128)),
              1: ((0, 128), (128, 256)),
              2: ((0, 128), (256, 384)),
              3: ((0, 128), (384, 512)),
              4: ((0, 128), (512, 640)),
              5: ((0, 128), (640, 768)),
              6: ((128, 256), (0, 128)),
              7: ((128, 256), (128, 256)),
              8: ((128, 256), (256, 384)),
              9: ((128, 256), (384, 512)),
              10: ((128, 256), (512, 640)),
              11: ((128, 256), (640, 768)),
              12: ((256, 384), (0, 128)),
              13: ((256, 384), (128, 256)),
              14: ((256, 384), (256, 384)),
              15: ((256, 384), (384, 512)),
              16: ((256, 384), (512, 640)),
              17: ((256, 384), (640, 768))}

def get_full_images(date, 
                    data_path='/scratch/snx3000/acarpent/HelioMontDataset/TestSet/KI/',
                    patches_idx=np.arange(18)):
    
    full_maps = np.empty((100, 128*3, 128*6))*np.nan
    patches_lst = []
    starting_idx_lst = []
    starting_idx_lst = set(np.arange(100))

    for p in patches_idx:
        patch = open_pkl(data_path+date+'_'+str(p)+'.pkl')
        maps = 2 * ((patch['ki_maps'] - 0.05) / (1.2 - 0.05)) - 1
        full_maps[:len(maps), patch_dict[p][0][0]:patch_dict[p][0][1],
                  patch_dict[p][1][0]:patch_dict[p][1][1]] = maps
        starting_idx_lst = starting_idx_lst.intersection(set(patch['starting_idx']))


    time = patch['time']
    full_maps = full_maps[:len(time)]
    x = ~np.isnan(full_maps).all(axis=(0, 2))
    full_maps = full_maps[:, x]
    y = ~np.isnan(full_maps).all(axis=(0, 1))
    full_maps = full_maps[:, :, y]
    
    return full_maps, starting_idx_lst, time

def get_full_coordinates(data_path='/scratch/snx3000/acarpent/HelioMontDataset/CoordinateData/',
                         patches_idx=np.arange(18),
                         normalization=False):
    
    full_lat = np.empty((128*3, 128*6))*np.nan
    full_lon = np.empty((128*3, 128*6))*np.nan
    full_alt = np.empty((128*3, 128*6))*np.nan
    for p in patches_idx:
        lat = open_pkl(data_path+str(p)+'_lat.pkl')
        lon = open_pkl(data_path+str(p)+'_lon.pkl')
        alt = open_pkl(data_path+str(p)+'_alt.pkl')
        full_lat[patch_dict[p][0][0]:patch_dict[p][0][1],
                  patch_dict[p][1][0]:patch_dict[p][1][1]] = lat
        full_lon[patch_dict[p][0][0]:patch_dict[p][0][1],
                  patch_dict[p][1][0]:patch_dict[p][1][1]] = lon
        full_alt[patch_dict[p][0][0]:patch_dict[p][0][1],
                  patch_dict[p][1][0]:patch_dict[p][1][1]] = alt
    
    x = ~np.isnan(full_lat).all(axis=(0))
    full_lat = full_lat[:, x]
    full_lon = full_lon[:, x]
    full_alt = full_alt[:, x]
    
    y = ~np.isnan(full_lat).all(axis=(1))
    full_lat = full_lat[y, :]
    full_lon = full_lon[y, :]
    full_alt = full_alt[y, :]

    if normalization:
        full_lon = 2 * ((full_lon - 0) / (90 - 0)) - 1
        full_lat = 2 * ((full_lat - 0) / (90 - 0)) - 1
        full_alt = 2 * ((full_alt - (-13)) / (4294 - 0)) - 1
    return full_lat, full_lon, full_alt


def compute_prob(arr, thresh, mean=True):
    x = arr.copy()
    x[x<thresh] = 0
    x[x>=thresh] = 1
    if mean:
        return np.nanmean(x, axis=0)
    else:
        return x


def remap(x, max_value=1.2, min_value=0.05):
    return ((x+1)/2)*(max_value-min_value) + min_value


def nonparametric_cdf_transform(initial_array, target_array, alpha):
    # flatten the arrays
    arrayshape = initial_array.shape
    target_array = target_array.flatten()
    initial_array = initial_array.flatten()
    # extra_array = extra_array.flatten()

    # rank target values
    order = target_array.argsort()
    target_ranked = target_array[order]

    # rank initial values order
    orderin = initial_array.argsort()
    ranks = np.empty(len(initial_array), int)
    ranks[orderin] = np.arange(len(initial_array))

    # # rank extra array
    orderex = initial_array.argsort()
    extra_ranked = initial_array[orderex]

    # get ranked values from target and rearrange with the initial order
    ranked = alpha*extra_ranked + (1-alpha)*target_ranked
    output_array = ranked[ranks]

    # reshape to the original array dimensions
    output_array = output_array.reshape(arrayshape)
    return output_array