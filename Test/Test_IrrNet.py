from validation_utils import get_diffusion_model
from Benchmark.IrradianceNet import ConvLSTM_patch, IrradianceNet
import numpy as np
from utils import save_pkl, open_pkl, get_full_images, get_full_coordinates, remap
import torch
import os
import sys
from yaml import load, Loader

print(sys.argv)
start = int(sys.argv[1])
end = int(sys.argv[2])
test_name = sys.argv[3]
model_config_path = sys.argv[4]
model_path = sys.argv[5]

def interpolate_yhat(yhat):
    yhat = yhat.detach()
    yhat[:, 125:131] = np.nan
    yhat[:, :, 125:131] = np.nan
    
    # rows
    for t in range(yhat.shape[0]):
        row_start_vals = yhat[t][124]
        row_end_vals = yhat[t][131]
        diff_interpolate = (row_start_vals - row_end_vals) / 7
        diff_interpolate = diff_interpolate.unsqueeze(0)  # .repeat(6, 1)
        diff_interpolate = diff_interpolate.repeat(6, 1)
        vals = np.arange(1, 7)
        vals = vals[np.newaxis, :]
        vals = np.repeat(vals, diff_interpolate.shape[1], axis=0)

        interpol_values = diff_interpolate.detach() * vals.T
        interpol_values = row_start_vals.unsqueeze(0).repeat(6, 1) - interpol_values
        yhat[t, 125:131] = interpol_values
        
        col_start_vals = yhat[t][:, 124]
        col_end_vals = yhat[t][:, 131]
        diff_interpolate = (col_start_vals - col_end_vals) / 7
        diff_interpolate = diff_interpolate.unsqueeze(0)  # .repeat(6, 1)
        diff_interpolate = diff_interpolate.repeat(6, 1)
        vals = np.arange(1, 7)
        vals = vals[np.newaxis, :]
        vals = np.repeat(vals, diff_interpolate.shape[1], axis=0)

        interpol_values = diff_interpolate.detach() * vals.T
        interpol_values = col_start_vals.unsqueeze(0).repeat(6, 1) - interpol_values
        yhat[t, :, 125:131] = interpol_values.T
    return yhat

def main():
    nowcast_net = ConvLSTM_patch(in_chan=1, image_size=128, device='cuda', seq_len=8)
    irradiance_net = IrradianceNet(nowcast_net,
                                   opt_patience=5).to('cuda')
    
    checkpoint = torch.load(model_path)
    irradiance_net.load_state_dict(checkpoint['state_dict'])
    model_config = open_pkl(model_config_path)
    model_id = model_config['ID']

    with open('/scratch/snx3000/acarpent/Test_Results/{}/config.yml'.format(test_name),
              'r') as o:
        test_config = load(o, Loader)

    data_path='/scratch/snx3000/acarpent/HelioMontDataset/{}/KI/'.format(test_config['dataset_name'])
    n_ens = test_config['n_ens']
    ddim_steps = test_config['ddim_steps']
    x_max = test_config['x_max']
    x_min = test_config['x_min']
    y_max = test_config['y_max']
    y_min = test_config['y_min']
    patches_idx = test_config['patches_idx']

    date_idx_dict = open_pkl('/scratch/snx3000/acarpent/Test_Results/{}/Test_date_idx.pkl'.format(test_name))
    test_days = list(date_idx_dict.keys())
    print(test_days)
    forecast_dict = {}

    for date in test_days[start:end]:
        full_maps, idx_lst, t = get_full_images(date, data_path=data_path, patches_idx=patches_idx)
        # idx = np.random.choice(list(idx_lst), replace=False, size=n_per_day)
        idx = date_idx_dict[date]
        print(date)
        for i in idx:
            x = torch.Tensor(full_maps[i:i+4, y_min:y_max, x_min:x_max])
            y = torch.Tensor(full_maps[i+4:i+12, y_min:y_max, x_min:x_max])
            x,y = x.reshape(1,1,*x.shape).to('cuda'), y.reshape(1,1,*y.shape).to('cuda')

            yhat = torch.zeros((8, 256, 256))
            for x_i in [0, 128]:
                for y_i in [0, 128]:
                    yhat[:, x_i:x_i+128, y_i:y_i+128] = irradiance_net(x[:, :, :, x_i:x_i+128, y_i:y_i+128]).detach()
            
            yhat = interpolate_yhat(yhat)
            yhat[yhat<-1] = -1
            yhat[yhat>1] = 1
            forecast_dict[t[i]] = np.array(yhat.numpy()).astype(np.float32)
        save_pkl('/scratch/snx3000/acarpent/Test_Results/{}/{}-forecast_dict_{}.pkl'.format(test_name, model_id, date), forecast_dict)
        forecast_dict = {}
    print('###################################### SAVED ######################################')

if __name__ == '__main__':
    main()