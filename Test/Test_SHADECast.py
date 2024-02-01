from validation_utils import get_diffusion_model
from SHADECast.Models.Sampler.PLMS import PLMSSampler
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

def main():
    ldm, model_config = get_diffusion_model(model_config_path,
                                            model_path)
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
    ldm = ldm.to('cuda')
    sampler = PLMSSampler(ldm, verbose=False)
    forecast_dict = {}
   
    print('Testing started')

    for date in test_days[start:end]:
        full_maps, idx_lst, t = get_full_images(date, data_path=data_path, patches_idx=patches_idx)
        # idx = np.random.choice(list(idx_lst), replace=False, size=n_per_day)
        idx = date_idx_dict[date]
        print(date)
        for i in idx:
            x = torch.Tensor(full_maps[i:i+4, y_min:y_max, x_min:x_max])
            y = torch.Tensor(full_maps[i+4:i+12, y_min:y_max, x_min:x_max])
            x,y = x.reshape(1,1,*x.shape).to('cuda'), y.reshape(1,1,*y.shape).to('cuda')
            # enc_x, _ = ldm.autoencoder.encode(x)
            enc_y, _ = ldm.autoencoder.encode(y)
            x = torch.cat([x for _ in range(n_ens)]).to('cuda')
            cond = ldm.context_encoder(x)
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_ens,
                                             shape=tuple(enc_y.shape[1:]),
                                             verbose=False,
                                             eta=0.)

            yhat = ldm.autoencoder.decode(samples_ddim.to('cuda'))
            yhat = yhat.to('cpu').detach().numpy()[:,0]
            yhat[yhat<-1] = -1
            yhat[yhat>1] = 1
            print(yhat.shape)
            # ens_members.append(yhat_)

            forecast_dict[t[i]] = np.array(yhat).astype(np.float32)
        save_pkl('/scratch/snx3000/acarpent/Test_Results/{}/{}_{}-ddim_forecast_dict_{}.pkl'.format(test_name, model_id, ddim_steps, date), forecast_dict)
        forecast_dict = {}
        print('###################################### SAVED ######################################')

if __name__ == '__main__':
    main()