from Blocks.AFNO import AFNOBlock3d
from Models.UNet.utils import PositionalEncoding, DoubleConv, Up, Down
import torch
from torch import nn
from Blocks.AFNO import AFNOCrossAttentionBlock3d


class AFNOUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 32,
            context_channels: int = 32,
            out_channels: int = 32,
            noise_steps: int = 1000,
            time_dim: int = 256,
    ):
        super(AFNOUNet, self).__init__()

        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)

        self.concat_afnoblock = AFNOCrossAttentionBlock3d(in_channels,
                                                          context_channels,
                                                          out_dim=in_channels+context_channels,
                                                          mlp_ratio=4.,
                                                          act_layer=nn.GELU,
                                                          norm_layer=nn.LayerNorm,
                                                          data_format="channels_first")
        in_channels = in_channels + context_channels

        self.down1 = Down(in_channels, in_channels * 2, emb_dim=time_dim)
        self.afnoblock1 = AFNOBlock3d(in_channels * 2, data_format="channels_first")
        self.down2 = Down(in_channels * 2, in_channels * 4, emb_dim=time_dim)
        self.afnoblock2 = AFNOBlock3d(in_channels * 4, data_format="channels_first")
        self.down3 = Down(in_channels * 4, in_channels * 4, emb_dim=time_dim)
        self.afnoblock3 = AFNOBlock3d(in_channels * 4, data_format="channels_first")

        self.bottleneck1 = DoubleConv(in_channels * 4, in_channels * 8)
        self.bottleneck2 = DoubleConv(in_channels * 8, in_channels * 8)
        self.bottleneck3 = DoubleConv(in_channels * 8, in_channels * 4)

        self.up1 = Up(in_channels * 8, in_channels * 2, emb_dim=time_dim)
        self.afnoblock4 = AFNOBlock3d(in_channels * 2, data_format="channels_first")
        self.up2 = Up(in_channels * 4, in_channels, emb_dim=time_dim)
        self.afnoblock5 = AFNOBlock3d(in_channels, data_format="channels_first")
        self.up3 = Up(in_channels * 2, in_channels, emb_dim=time_dim)
        self.afnoblock6 = AFNOBlock3d(in_channels, data_format="channels_first")
        self.out_conv1 = DoubleConv(in_channels=in_channels, out_channels=in_channels)
        self.out_conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """Forward pass with image tensor and timestep reduce noise.

        Args:
            x: Image tensor of shape, [batch_size, channels, height, width].
            t: Time step defined as long integer. If batch size is 4, noise step 500, then random timesteps t = [10, 26, 460, 231].
            :param t:
            :param c:
        """
        t = self.pos_encoding(t)

        # concatenation of context with original input

        x1 = self.concat_afnoblock(x, c)
        x2 = self.down1(x1, t)
        x2 = self.afnoblock1(x2)
        x3 = self.down2(x2, t)
        x3 = self.afnoblock2(x3)
        x4 = self.down3(x3, t)
        x4 = self.afnoblock3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x = self.up1(x4, x3, t)
        x = self.afnoblock4(x)
        x = self.up2(x, x2, t)
        x = self.afnoblock5(x)
        x = self.up3(x, x1, t)
        # x = self.afnoblock6(x)
        x = self.out_conv1(x)
        return self.out_conv2(x)
