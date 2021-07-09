import torch.nn as nn
import torch.nn.functional as F

from src.utils.args import args
from src.modules.nn_layers import *
from src.modules.distributions import n_embenddings


class q_z(nn.Module):
    """ Encoder q(z|x)
    """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        nc_in = input_shape[0]
        nc_out = 2 * output_shape[0]
        self.output_shape = output_shape
        self.nh_out, self.nw_out = output_shape[1], output_shape[2]

        self.core_nn = nn.Sequential(
            Encoder(
                in_channels=nc_in,
                out_channels=nc_out,
                #growth_rate=nc_out//2,
                steps=5,
                scale_factor=3)
        )

        flat_dim = 256*self.nh_out*self.nw_out
        #self.mean_block = nn.Linear(128, 128)
        #self.logvar_block = nn.Linear(128, 128)

    def forward(self, input):
        mu, logvar = self.core_nn(input).chunk(2, 1)
        #mu = self.mean_block(h.reshape(h.shape[0], -1)).reshape([input.shape[0]] + list(self.output_shape))
        #logvar = self.logvar_block(h.reshape(h.shape[0], -1)).reshape([input.shape[0]] + list(self.output_shape))
        return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)


class p_x(nn.Module):
    """ Decoder p(x|z)
    """
    def __init__(self, output_shape, input_shape):
        super().__init__()
        self.nc_in, self.nh_in, self.nw_in = input_shape[0], input_shape[1], input_shape[2]
        nc_out = output_shape[0]#n_embenddings(output_shape[0])

        self.core_nn = nn.Sequential(
            Decoder(
                in_channels=self.nc_in,
                out_channels=nc_out,
                #growth_rate=128,
                steps=5,
                scale_factor=3)
        )


    def forward(self, input):
        return self.core_nn(input)


if __name__ == "__main__":
    pass
