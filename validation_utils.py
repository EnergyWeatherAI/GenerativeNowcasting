from Dataset.dataset import KIDataset
from torch.utils.data import DataLoader
from SHADECast.Models.Nowcaster.Nowcast import AFNONowcastNetCascade, Nowcaster, AFNONowcastNet, CAFNONowcastNetCascade, ContextEncoder

from SHADECast.Models.VAE.VariationalAutoEncoder import VAE, Encoder, Decoder
from SHADECast.Models.UNet.UNet import UNetModel
from SHADECast.Models.Diffusion.DiffusionModel import LatentDiffusion
from utils import open_pkl, save_pkl

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
    return dataloader, dataset


def get_diffusion_model(config_path, ldm_path):
    
    config = open_pkl(config_path)
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
    vae = VAE.load_from_checkpoint(vae_config['path'],
                                   encoder=encoder, decoder=decoder,
                                   opt_patience=5)

    print('VAE built')

    nowcaster_config = config['Nowcaster']
    if nowcaster_config['path'] is None:
        nowcast_net = AFNONowcastNet(vae,
                                    train_autoenc=False,
                                    embed_dim=nowcaster_config['embed_dim'],
                                    embed_dim_out=nowcaster_config['embed_dim'],
                                    analysis_depth=nowcaster_config['analysis_depth'],
                                    forecast_depth=nowcaster_config['forecast_depth'],
                                    input_steps=nowcaster_config['input_steps'],
                                    output_steps=nowcaster_config['output_steps'],
        )
    else:
        nowcast_net = AFNONowcastNet(vae,
                                    train_autoenc=False,
                                    embed_dim=nowcaster_config['embed_dim'],
                                    embed_dim_out=nowcaster_config['embed_dim'],
                                    analysis_depth=nowcaster_config['analysis_depth'],
                                    forecast_depth=nowcaster_config['forecast_depth'],
                                    input_steps=nowcaster_config['input_steps'],
                                    output_steps=nowcaster_config['output_steps'],
        )
        nowcaster = Nowcaster.load_from_checkpoint(nowcaster_config['path'], nowcast_net=nowcast_net,
                                                opt_patience=nowcaster_config['opt_patience'],
                                                loss_type=nowcaster_config['loss_type'])
        nowcast_net = nowcaster.nowcast_net
    
    cascade_net = AFNONowcastNetCascade(nowcast_net=nowcast_net, 
                                        cascade_depth=nowcaster_config['cascade_depth'])
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

    ldm = LatentDiffusion.load_from_checkpoint(ldm_path,
                                               model=denoiser,
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
                                               opt_patience=diffusion_config['opt_patience']
                              )
    return ldm, config