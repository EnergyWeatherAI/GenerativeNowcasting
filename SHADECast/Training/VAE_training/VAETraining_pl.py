import os
from torchinfo import summary
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from yaml import load, Loader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning import seed_everything
from Models.VAE.VariationalAutoEncoder import Encoder, Decoder, VAE
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
                   shuffle=True):
    dataset = KIDataset(data_path=data_path,
                        n=n,
                        min=min,
                        max=max,
                        length=length,
                        norm_method=norm_method,
                        coordinate_data_path=coordinate_data_path,
                        return_all=False)
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader


def train(config):
    num_nodes = int(os.environ['SLURM_NNODES'])
    rank = int(os.environ['SLURM_NODEID'])
    print(rank, num_nodes)

    ID = config['ID']
    save_pkl(config['Checkpoint']['dirpath'] + ID + '_config.pkl', config),
    print(config)

    encoder_config = config['Encoder']
    encoder = Encoder(in_dim=encoder_config['in_dim'],
                      levels=encoder_config['levels'],
                      min_ch=encoder_config['min_ch'],
                      max_ch=encoder_config['max_ch'])
    print('Encoder built')

    decoder_config = config['Decoder']
    decoder = Decoder(in_dim=decoder_config['in_dim'],
                      levels=decoder_config['levels'],
                      min_ch=decoder_config['min_ch'],
                      max_ch=decoder_config['max_ch'])
    print('Decoder built')

    vae_config = config['VAE']
    vae = VAE(encoder,
              decoder,
              kl_weight=vae_config['kl_weight'],
              encoded_channels=encoder_config['max_ch'],
              hidden_width=vae_config['hidden_width'],
              opt_patience=vae_config['opt_patience'])
    print('All models built')

    batch_size = config['Dataset']['batch_size']
    n_steps = config['Dataset']['n_steps']
    if rank == 0:
        summary(vae, input_size=(batch_size, 1, n_steps, 128, 128))

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
        deterministic=True
    )

    data_config = config['Dataset']
    train_dataloader = get_dataloader(data_path=data_config['data_path'] + 'TrainingSet/KI/',
                                      coordinate_data_path=data_config['data_path']+'CoordinateData/',
                                      n=data_config['n_steps'],
                                      min=data_config['min'],
                                      max=data_config['max'],
                                      length=data_config['train_length'],
                                      num_workers=data_config['num_workers'],
                                      norm_method=data_config['norm_method'],
                                      batch_size=data_config['batch_size'],
                                      shuffle=True)

    val_dataloader = get_dataloader(data_path=data_config['data_path'] + 'ValidationSet/KI/',
                                    coordinate_data_path=data_config['data_path'] + 'CoordinateData/',
                                    n=data_config['n_steps'],
                                    min=data_config['min'],
                                    max=data_config['max'],
                                    length=data_config['val_length'],
                                    num_workers=data_config['num_workers'],
                                    norm_method=data_config['norm_method'],
                                    batch_size=data_config['batch_size'],
                                    shuffle=False)
    print('Training started')

    resume_training = tr_config['resume_training']
    if resume_training is None:
        trainer.fit(vae, train_dataloader, val_dataloader)
    else:
        trainer.fit(vae, train_dataloader, val_dataloader,
                    ckpt_path=resume_training)


if __name__ == '__main__':
    with open('/scratch/snx3000/acarpent/GenerativeNowcasting/Training/VAE_training/VAEtrainingconf.yml', 'r') as o:
        config = load(o, Loader)

    # seed_everything(0, workers=0)
    train(config)
