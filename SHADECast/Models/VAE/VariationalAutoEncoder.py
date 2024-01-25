import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from SHADECast.Blocks.ResBlock3D import ResBlock3D
from SHADECast.Blocks.AFNO import AFNOBlock3d
from utils import sample_from_standard_normal, kl_from_standard_normal


class Encoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64, max_ch=64, afno=False):
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            res_kernel_size = (3, 3, 3) if i == 0 else (1, 3, 3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            if afno:
                afno_block = AFNOBlock3d(
                    out_channels, 
                    mlp_ratio=2*i, 
                    num_blocks=16, 
                    data_format='channels_first')
                sequence.append(afno_block)
            downsample = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=(2, 2, 2), stride=(2, 2, 2))
            sequence.append(downsample)

        super().__init__(*sequence)


class Decoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64, max_ch=64, afno=False):
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        for i in reversed(list(range(levels))):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])
            upsample = nn.ConvTranspose3d(in_channels, in_channels,
                                          kernel_size=(2, 2, 2), stride=(2, 2, 2))
            sequence.append(upsample)
            
            if afno:
                afno_block = AFNOBlock3d(
                    in_channels, 
                    mlp_ratio=2*i,
                    num_blocks=16, 
                    data_format='channels_first')
                sequence.append(afno_block)
            
            res_kernel_size = (3, 3, 3) if (i == 0) else (1, 3, 3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)

        super().__init__(*sequence)


class VAE(pl.LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 kl_weight,
                 encoded_channels,
                 hidden_width,
                 opt_patience,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.opt_patience = opt_patience
        self.to_moments = nn.Conv3d(encoded_channels, 2 * hidden_width,
                                    kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
                                    kernel_size=1)

        self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return mean, log_var

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        (mean, log_var) = self.encode(x)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return dec, mean, log_var

    def _loss(self, batch):
        x = batch

        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (x - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)

        total_loss = (1 - self.kl_weight) * rec_loss + self.kl_weight * kl_loss

        return total_loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log('train_loss', loss, **log_params)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3,
                                      betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_patience, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",
                "frequency": 1,
            },
        }
