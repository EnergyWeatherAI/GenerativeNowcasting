import collections
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from SHADECast.Blocks.attention import TemporalTransformer
from SHADECast.Blocks.AFNO import AFNOBlock3d, AFNOCrossAttentionBlock3d
from SHADECast.Blocks.ResBlock3D import ResBlock3D, ResBlock2D
import numpy as np


class ConditionedNowcaster(pl.LightningModule):
    def __init__(self, nowcast_net, opt_patience, loss_type='l1'):
        super().__init__()
        self.nowcast_net = nowcast_net
        self.opt_patience = opt_patience
        self.loss_type = loss_type

    def forward(self, x, c, t):
        return self.nowcast_net(x, c, t)

    def _loss(self, batch):
        x, y, t, c = batch

        if self.loss_type == 'l1':
            y_pred = self.forward(x, c, t)
            return (y - y_pred).abs().mean()

        elif self.loss_type == 'l2':
            y_pred = self.forward(x, c, t)
            return (y - y_pred).square().mean()

        elif self.loss_type == 'latent':
            y, _ = self.nowcast_net.autoencoder.encode(y)
            x = self.nowcast_net.latent_forward(x, c, t)
            y_pred = self.nowcast_net.out_proj(x)
            return (y - y_pred).abs().mean()
        else:
            AssertionError('Loss type must be "l1" or "l2" or "latent"')

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


class CoordEncoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64, max_ch=64):
        self.ch = max_ch
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            res_kernel_size = (3, 3)
            res_block = ResBlock2D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            downsample = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=(2, 2), stride=(2, 2))
            sequence.append(downsample)

        super().__init__(*sequence)


class TimeEncoder(nn.Module):
    def __init__(self, in_dim=1, out_dim=128, levels=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis = nn.LSTM(input_size=in_dim, hidden_size=out_dim, num_layers=levels)
        self.time_encoder = nn.Conv1d(out_dim, out_dim, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.analysis(x)[0].permute((0, 2, 1))
        x = self.time_encoder(x).permute((0, 2, 1))
        return x


class AFNONowcastNetBase(nn.Module):
    def __init__(
            self,
            autoencoder,
            coord_encoder,
            time_encoder,
            embed_dim=128,
            embed_dim_out=None,
            analysis_depth=4,
            forecast_depth=4,
            input_steps=1,
            output_steps=2,
            train_autoenc=False
    ):
        super().__init__()

        self.train_autoenc = train_autoenc
        self.coord_encoder = coord_encoder
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_steps = output_steps
        self.input_steps = input_steps

        # encoding + analysis for each input
        ae = autoencoder.requires_grad_(train_autoenc)
        self.autoencoder = ae
        self.proj = nn.Conv3d(ae.hidden_width, embed_dim, kernel_size=1)
        self.afno_cross_attention_1 = AFNOCrossAttentionBlock3d(dim=embed_dim,
                                                                context_dim=coord_encoder.ch,
                                                                data_format="channels_first")
        self.afno_cross_attention_2 = AFNOCrossAttentionBlock3d(dim=embed_dim,
                                                                context_dim=coord_encoder.ch,
                                                                data_format="channels_last")
        self.time_encoder = time_encoder
        self.analysis = nn.Sequential(
            *(AFNOBlock3d(embed_dim) for _ in range(analysis_depth))
        )

        # temporal transformer
        self.use_temporal_transformer = input_steps != output_steps
        if self.use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(embed_dim)

        # forecast
        self.forecast = nn.Sequential(
            *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
        )

    def forward(self, x, c, sza):
        c = self.coord_encoder(c)
        c = c.reshape(*c.shape[:2], 1, *c.shape[2:])
        sza = self.time_encoder(sza)
        sza = sza.reshape(*sza.shape[:2], 1, 1, sza.shape[-1])
        x = self.autoencoder.encode(x)[0]
        x = self.proj(x)
        x = self.afno_cross_attention_1(x, c)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.analysis(x)
        if self.use_temporal_transformer:
            # add positional encoding
            expand_shape = x.shape[:1] + (-1,) + x.shape[2:]
            in_sza = sza[:, :self.input_steps, :].expand(*expand_shape)
            out_sza = sza[:, self.input_steps:, :].expand(*expand_shape)
            x = x + in_sza

            x = self.temporal_transformer(out_sza, x)

        c = c.permute(0, 2, 3, 4, 1).expand(*x.shape[:-1], -1)
        x = self.afno_cross_attention_2(x, c)
        x = self.forecast(x)

        return x.permute(0, 4, 1, 2, 3)  # to channels-first order


class CAFNONowcastNet(AFNONowcastNetBase):
    def __init__(self, autoencoder, **kwargs):
        super().__init__(autoencoder, **kwargs)

        self.output_autoencoder = autoencoder.requires_grad_(
            self.train_autoenc)
        self.out_proj = nn.Conv3d(
            self.embed_dim_out, autoencoder.hidden_width, kernel_size=1
        )

    def latent_forward(self, x, c, t):
        x = super().forward(x, c, t)
        return x

    def forward(self, x, c, t):
        x = self.latent_forward(x, c, t)
        x = self.out_proj(x)
        return self.output_autoencoder.decode(x)


class CAFNONowcastNetCascade(nn.Module):
    def __init__(self,
                 nowcast_net,
                 cascade_depth=4,
                 train_net=False):
        super().__init__()
        self.cascade_depth = cascade_depth
        self.nowcast_net = nowcast_net
        for p in self.nowcast_net.parameters():
            p.requires_grad = train_net
        # self.nowcast_net.requires_grad(train_net)
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

    def forward(self, x, c, t):
        x = self.nowcast_net.latent_forward(x, c, t)
        img_shape = tuple(x.shape[-2:])
        cascade = {img_shape: x}
        for i in range(self.cascade_depth - 1):
            x = F.avg_pool3d(x, (1, 2, 2))
            x = self.resnet[i](x)
            img_shape = tuple(x.shape[-2:])
            cascade[img_shape] = x
        return cascade
