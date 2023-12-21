import os
from torchinfo import summary
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from yaml import load, Loader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning import seed_everything
from Models.VAE.VariationalAutoEncoder import Encoder, Decoder, VAE
from Models.Nowcaster.Nowcast import AFNONowcastNet, Nowcaster
from Dataset.dataset import KIDataset
from utils import save_pkl


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
                   validation=False):
    dataset = KIDataset(data_path=data_path,
                        n=n,
                        min=min,
                        max=max,
                        length=length,
                        norm_method=norm_method,
                        coordinate_data_path=coordinate_data_path,
                        return_all=False,
                        forecast=True,
                        validation=validation)
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
    if vae_config['path'] is not None:
        vae = VAE.load_from_checkpoint(vae_config['path'],
                                       encoder=encoder, decoder=decoder)
        train_autoencoder = False
    else:
        vae = VAE(encoder,
                  decoder,
                  kl_weight=vae_config['kl_weight'],
                  encoded_channels=encoder_config['max_ch'],
                  hidden_width=vae_config['hidden_width'])
        train_autoencoder = True
    if rank == 0:
        print('VAE built')

    nowcaster_config = config['Nowcaster']
    if rank == 0:
        print(nowcaster_config)

    nowcast_net = AFNONowcastNet(vae,
                                 train_autoenc=train_autoencoder,
                                 embed_dim=nowcaster_config['embed_dim'],
                                 embed_dim_out=nowcaster_config['embed_dim'],
                                 analysis_depth=nowcaster_config['analysis_depth'],
                                 forecast_depth=nowcaster_config['forecast_depth'],
                                 input_steps=nowcaster_config['input_steps'],
                                 output_steps=nowcaster_config['output_steps'])
    nowcaster = Nowcaster(nowcast_net=nowcast_net,
                          opt_patience=nowcaster_config['opt_patience'],
                          loss_type=nowcaster_config['loss_type'])

    if rank == 0:
        print('All models built')
        summary(nowcaster)

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
        accumulate_grad_batches=tr_config['accumulate_grad_batches']
        # deterministic=False
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
                                    validation=True)
    if rank == 0:
        print('Training started')
    resume_training = tr_config['resume_training']
    if resume_training is None:
        trainer.fit(nowcaster, train_dataloader, val_dataloader)
    else:
        trainer.fit(nowcaster, train_dataloader, val_dataloader,
                    ckpt_path=resume_training)


if __name__ == '__main__':
    with open('/scratch/snx3000/acarpent/GenerativeNowcasting/Training/Nowcast_training/Nowcastertrainingconf.yml',
              'r') as o:
        config = load(o, Loader)

    # seed_everything(0, workers=0)

    train(config)
