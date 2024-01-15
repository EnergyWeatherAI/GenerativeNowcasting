import collections
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from Blocks.attention import TemporalTransformer, positional_encoding
from Blocks.AFNO import AFNOBlock3d, AFNOCrossAttentionBlock3d
from Blocks.ResBlock3D import ResBlock3D
import numpy as np


class HRNowcaster(pl.LightningModule):
    def __init__(self, nowcast_net, opt_patience, loss_type='l1'):
        super().__init__()
        self.nowcast_net = nowcast_net
        self.opt_patience = opt_patience
        self.loss_type = loss_type

    def forward(self, x):
        return self.nowcast_net(x)

    def _loss(self, batch):
        x, y = batch

        if self.loss_type == 'l1':
            y_pred = self.forward(x)
            return (y - y_pred).abs().mean()

        elif self.loss_type == 'l2':
            y_pred = self.forward(x)
            return (y - y_pred).square().mean()

        elif self.loss_type == 'latent':
            y, _ = self.nowcast_net.autoencoder.encode(y)
            x = self.nowcast_net.latent_forward(x)
            y_pred = self.nowcast_net.out_proj(x)
            return (y - y_pred).abs().mean()
        else:
            AssertionError('Loss type must be "l1" or "l2"')

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log('train_loss', loss, **log_params)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}_loss", loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_patience, factor=0.25, verbose=True
        )

        optimizer_spec = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
        return optimizer_spec


class HRAFNONowcastNetBase(nn.Module):
    def __init__(
            self,
            autoencoder,
            embed_dim=128,
            embed_dim_out=None,
            analysis_depth=4,
            forecast_depth=4,
            input_steps=1,
            output_steps=2,
            train_autoenc=False,
            afno_res_mult=1
    ):
        super().__init__()

        self.train_autoenc = train_autoenc
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.afno_res_mult = afno_res_mult
        # encoding + analysis for each input
        ae = autoencoder.requires_grad_(train_autoenc)
        self.autoencoder = ae

        self.proj = nn.Conv3d(ae.hidden_width, embed_dim, kernel_size=1)

        self.analysis = nn.Sequential(*(AFNOBlock3d(embed_dim, afno_res_mult=self.afno_res_mult),
                *(AFNOBlock3d(embed_dim) for _ in range(analysis_depth-1)))
            )

        # temporal transformer
        self.use_temporal_transformer = input_steps != output_steps
        if self.use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(embed_dim)

        # forecast
        self.forecast = nn.Sequential(
            *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
        )

    def add_pos_enc(self, x, t):
        if t.shape[1] != x.shape[1]:
            # this can happen if x has been compressed
            # by the autoencoder in the time dimension
            ds_factor = t.shape[1] // x.shape[1]
            t = F.avg_pool1d(t.unsqueeze(1), ds_factor)[:, 0, :]

        pos_enc = positional_encoding(t, x.shape[-1], add_dims=(2, 3))
        return x + pos_enc

    def forward(self, x):
        # (x, t_relative) = list(zip(*x))

        # encoding + analysis for each input
        # def process_input(i):
        x = self.autoencoder.encode(x)[0]
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.analysis(x)
        if self.use_temporal_transformer:
            # add positional encoding
            t = torch.arange(0, self.input_steps, device=x.device)
            expand_shape = x.shape[:1] + (-1,) + x.shape[2:]
            pos_enc_output = positional_encoding(
                t,
                self.embed_dim, add_dims=(0, 2, 3)
            )
            pe_out = pos_enc_output.expand(*expand_shape)
            x = x + pe_out

            # transform to output shape and coordinates
            pos_enc_output = positional_encoding(
                torch.arange(self.input_steps, self.output_steps + 1, device=x.device),
                self.embed_dim, add_dims=(0, 2, 3)
            )
            pe_out = pos_enc_output.expand(*expand_shape)
            x = self.temporal_transformer(pe_out, x)

        x = self.forecast(x)
        return x.permute(0, 4, 1, 2, 3)  # to channels-first order


class HRAFNONowcastNet(HRAFNONowcastNetBase):
    def __init__(self, autoencoder, **kwargs):
        super().__init__(autoencoder, **kwargs)

        self.output_autoencoder = autoencoder.requires_grad_(
            self.train_autoenc)
        self.out_proj = nn.Conv3d(
            self.embed_dim_out, autoencoder.hidden_width, kernel_size=1
        )

    def latent_forward(self, x):
        x = super().forward(x)
        return x

    def forward(self, x):
        x = self.latent_forward(x)
        x = self.out_proj(x)
        return self.output_autoencoder.decode(x)


class AFNONowcastNetCascade(nn.Module):
    def __init__(self,
                 nowcast_net,
                 cascade_depth=4,
                 train_net=False):
        super().__init__()
        self.cascade_depth = cascade_depth
        self.nowcast_net = nowcast_net
        for p in self.nowcast_net.parameters():
            p.requires_grad = train_net
        self.resnet = nn.ModuleList()
        ch = self.nowcast_net.embed_dim_out
        self.cascade_dims = [ch]
        for i in range(cascade_depth - 1):
            ch_out = 2 * ch
            self.cascade_dims.append(ch_out)
            self.resnet.append(
                ResBlock3D(ch, ch_out, kernel_size=(1, 3, 3), norm=None)
            )
            ch = ch_out

    def forward(self, x):
        x = self.nowcast_net.latent_forward(x)
        img_shape = tuple(x.shape[-2:])
        cascade = {img_shape: x}
        for i in range(self.cascade_depth - 1):
            x = F.avg_pool3d(x, (1, 2, 2))
            x = self.resnet[i](x)
            img_shape = tuple(x.shape[-2:])
            cascade[img_shape] = x
        return cascade