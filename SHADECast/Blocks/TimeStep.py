from abc import abstractmethod
import torch.nn as nn
from SHADECast.Blocks.AFNO import AFNOCrossAttentionBlock3d

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AFNOCrossAttentionBlock3d):
                img_shape = tuple(x.shape[-2:])
                x = layer(x, context[img_shape])
            else:
                x = layer(x)
        return x