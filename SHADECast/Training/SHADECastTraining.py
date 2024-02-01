import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from yaml import load, Loader
from torchinfo import summary
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import save_pkl
from Dataset.dataset import KIDataset
from Models.Nowcaster.Nowcast import AFNONowcastNetCascade, Nowcaster, AFNONowcastNet
from Models.VAE.VariationalAutoEncoder import VAE, Encoder, Decoder
from Models.UNet.UNet import UNetModel
from Models.Diffusion.DiffusionModel import LatentDiffusion


def get_dataloader(data_path,
                   coordinate_data_path,
                   n=12,
                   min=0.05,
                   max=1.2,
                   length=None,
                   norm_method='rescaling',
                   num_workers=24,
                   batch_size=64,
                   shuffle=True,
                   validation=False,
                   return_t=False):
    dataset = KIDataset(data_path=data_path,
                        n=n,
                        min=min,
                        max=max,
                        length=length,
                        norm_method=norm_method,
                        coordinate_data_path=coordinate_data_path,
                        return_all=False,
                        forecast=True,
                        validation=validation,
                        return_t=return_t)
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader


def train(config, distributed=True):
    if distributed:
        num_nodes = int(os.environ['SLURM_NNODES'])
        rank = int(os.environ['SLURM_NODEID'])
        print(rank, num_nodes)
    else:
        rank = 0
        num_nodes = 1

    ID = config['ID']
    save_pkl(config['Checkpoint']['dirpath'] + ID + '_config.pkl', config)
    if rank == 0:
        print(config)

    encoder_config = config['Encoder']
    encoder = Encoder(in_dim=encoder_config['in_dim'],
                      levels=encoder_config['levels'],
                      min_ch=encoder_config['min_ch'],
                      max_ch=encoder_config['max_ch'])
    if rank == 0:
        print('Encoder built')

    decoder_config = config['Decoder']
    decoder = Decoder(in_dim=decoder_config['in_dim'],
                      levels=decoder_config['levels'],
                      min_ch=decoder_config['min_ch'],
                      max_ch=decoder_config['max_ch'])
    if rank == 0:
        print('Decoder built')

    vae_config = config['VAE']
    vae = VAE.load_from_checkpoint(vae_config['path'],
                                   encoder=encoder, decoder=decoder,
                                   opt_patience=vae_config['opt_patience'])
    if rank == 0:
        print('VAE built')

    nowcaster_config = config['Nowcaster']
    print(nowcaster_config['path'])
    if nowcaster_config['path'] is None:
        nowcast_net = AFNONowcastNet(vae,
                                     train_autoenc=False,
                                     embed_dim=nowcaster_config['embed_dim'],
                                     embed_dim_out=nowcaster_config['embed_dim'],
                                     analysis_depth=nowcaster_config['analysis_depth'],
                                     forecast_depth=nowcaster_config['forecast_depth'],
                                     input_steps=nowcaster_config['input_steps'],
                                     output_steps=nowcaster_config['output_steps'],
                                    #  opt_patience=nowcaster_config['opt_patience'],
                                    #  loss_type=nowcaster_config['loss_type']
        )
        train_nowcast = True 
    else:
        nowcast_net = AFNONowcastNet(vae,
                                     train_autoenc=False,
                                     embed_dim=nowcaster_config['embed_dim'],
                                     embed_dim_out=nowcaster_config['embed_dim'],
                                     analysis_depth=nowcaster_config['analysis_depth'],
                                     forecast_depth=nowcaster_config['forecast_depth'],
                                     input_steps=nowcaster_config['input_steps'],
                                     output_steps=nowcaster_config['output_steps'],
                                    #  opt_patience=nowcaster_config['opt_patience'],
                                    #  loss_type=nowcaster_config['loss_type']
        )
        nowcaster = Nowcaster.load_from_checkpoint(nowcaster_config['path'], nowcast_net=nowcast_net,
                                                   opt_patience=nowcaster_config['opt_patience'],
                                                   loss_type=nowcaster_config['loss_type'])
        nowcast_net = nowcaster.nowcast_net
        train_nowcast = False
    
    print('Nowcaster built, train: ', nowcaster_config['path'])
    cascade_net = AFNONowcastNetCascade(nowcast_net=nowcast_net, 
                                        cascade_depth=nowcaster_config['cascade_depth'],
                                        train_net=train_nowcast)
    if rank == 0:
        summary(nowcast_net)
    # if nowcaster_config['path'] is not None:
    #     nowcaster = Nowcaster.load_from_checkpoint(nowcaster_config['path'],
    #                                                nowcast_net=nowcast_net,
    #                                                autoencoder=vae)
    if rank == 0:
        print('Nowcaster built')

    diffusion_config = config['Diffusion']
    denoiser = UNetModel(
        in_channels=vae.hidden_width,
        model_channels=diffusion_config['model_channels'],
        out_channels=vae.hidden_width,
        num_res_blocks=diffusion_config['num_res_blocks'],
        attention_resolutions=diffusion_config['attention_resolutions'],
        dims=diffusion_config['dims'],
        channel_mult=diffusion_config['channel_mult'],
        num_heads=8,
        num_timesteps=2,
        context_ch=cascade_net.cascade_dims)

    ldm = LatentDiffusion(model=denoiser,
                          autoencoder=vae,
                          context_encoder=cascade_net,
                          beta_schedule=diffusion_config['scheduler'],
                          loss_type="l2",
                          use_ema=diffusion_config['use_ema'],
                          lr_warmup=0,
                          linear_start=1e-4,
                          linear_end=2e-2,
                          cosine_s=8e-3,
                          parameterization='eps',
                          lr=diffusion_config['lr'],
                          timesteps=diffusion_config['noise_steps'],
                          opt_patience=diffusion_config['opt_patience'],
                          get_t=config['Dataset']['get_t'],
                          )
    if rank == 0:
        print('All models built')
        summary(ldm)

    ckpt_config = config['Checkpoint']
    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_config['monitor'],
        dirpath=ckpt_config['dirpath'],
        filename=ID + '_' + ckpt_config['filename'],
        save_top_k=ckpt_config['save_top_k'],
        every_n_epochs=ckpt_config['every_n_epochs']
    )

    early_stop_callback = EarlyStopping(monitor=ckpt_config['monitor'],
                                        patience=config['EarlyStopping']['patience'])

    tr_config = config['Trainer']
    trainer = pl.Trainer(
        default_root_dir=ckpt_config['dirpath'],
        accelerator=tr_config['accelerator'],
        devices=tr_config['devices'],
        num_nodes=num_nodes,
        max_epochs=tr_config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy=tr_config['strategy'],
        precision=tr_config['precision'],
        enable_progress_bar=(rank == 0),
        deterministic=False,
        accumulate_grad_batches=tr_config['accumulate_grad_batches']
    )
    if rank == 0:
        print('Trainer built')
    data_config = config['Dataset']
    train_dataloader = get_dataloader(data_path=data_config['data_path'] + 'TrainingSet/KI/',
                                      coordinate_data_path=data_config['data_path'] + 'CoordinateData/',
                                      n=data_config['n_in'] + data_config['n_out'],
                                      min=data_config['min'],
                                      max=data_config['max'],
                                      length=data_config['train_length'],
                                      num_workers=data_config['num_workers'],
                                      norm_method=data_config['norm_method'],
                                      batch_size=data_config['batch_size'],
                                      shuffle=True,
                                      validation=False)

    val_dataloader = get_dataloader(data_path=data_config['data_path'] + 'ValidationSet/KI/',
                                    coordinate_data_path=data_config['data_path'] + 'CoordinateData/',
                                    n=data_config['n_in'] + data_config['n_out'],
                                    min=data_config['min'],
                                    max=data_config['max'],
                                    length=data_config['val_length'],
                                    num_workers=data_config['num_workers'],
                                    norm_method=data_config['norm_method'],
                                    batch_size=data_config['batch_size'],
                                    shuffle=False,
                                    validation=True,
                                    return_t=data_config['get_t'])
    if rank == 0:
        print('Training started')
    resume_training = tr_config['resume_training']
    torch.cuda.empty_cache()
    if resume_training is None:
        trainer.fit(ldm, train_dataloader, val_dataloader)
    else:
        # if tr_config['resume_training'] is False:
        trainer.fit(ldm, train_dataloader, val_dataloader,
                    ckpt_path=resume_training)
        # else:



if __name__ == '__main__':
    with open('SHADECastTrainingconf.yml',
              'r') as o:
        config = load(o, Loader)
    seed = config['seed']
    if seed is not None:
        seed_everything(int(seed), workers=True)

    train(config)
