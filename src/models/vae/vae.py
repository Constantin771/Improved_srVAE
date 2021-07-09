from functools import partial

import numpy as np

import torch
import torch.nn as nn

from src.utils import args, get_shape
from src.modules import *
from robust_loss_pytorch import AdaptiveLossFunction
#from kornia.filters import gaussian_blur2d


# ----- NN Model Selection -----

if args.model == 'VAE':
    if args.network == 'densenet64':
        from .image_networks.densenet64 import *
    else:
        raise NotImplementedError("Please use 'densenet64' as 'network' argument.")


# ----- Variational AutoEncoder -----

class VAE(nn.Module):
    """
    Variational AutoEncoder.

    Authors:
    Ioannis Gatopoulos, 2020
    Constantin Stipnieks, 2021
    """
    def __init__(self, x_shape, prior=args.prior):
        super().__init__()
        self.x_shape = x_shape

        self.z_dim = args.z_dim
        self.z_shape = get_shape(self.z_dim)

        # p(z)
        self.p_z = globals()[prior](self.z_shape)

        # q(z | x)
        self.q_z = q_z(self.z_shape, self.x_shape)

        # p(x | z)
        self.p_x = p_x(self.x_shape, self.z_shape)

        # likelihood distribution
        flat_x_dim = x_shape[0]*x_shape[1]*x_shape[2]
        self.robust_loss = AdaptiveLossFunction(num_dims=flat_x_dim, float_dtype=np.float32, device=torch.cuda.current_device())
        #self.recon_loss = partial(dmol_loss, nc=self.x_shape[0])
        #self.sample_distribution = partial(sample_from_dmol, nc=self.x_shape[0])


    '''
    def initialize(self, dataloader):
        """ Data dependent init for weight normalization 
            (Automatically done during the first forward pass).
        """
        with torch.no_grad():
            x, _ = next(iter(dataloader))
            x = x.to(args.device)
            output = self.forward(x)
            self.calculate_elbo(x, output, args.batch_size/len(dataloader))
        return
    '''

    @staticmethod
    def reparameterize(z_mu, z_logvar):
        """ z ~ N(z| z_mu, z_logvar)
        """
        epsilon = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5 * z_logvar) * epsilon

    @torch.no_grad()
    def generate(self, n_samples=args.n_samples):
        # u ~ p(u)
        z = self.p_z.sample(n_samples=n_samples, device=args.device).to(args.device)
        # x ~ p(x| z)
        x_hat = self.p_x(z.contiguous())

        return x_hat

    @torch.no_grad()
    def reconstruct(self, x, **kwargs):
        x_hat = self.forward(x).get('x_hat')
        return x_hat

    def calculate_elbo(self, input, outputs, kl_weight=1e-3):
        # unpack variables
        x, x_hat = input, outputs.get('x_hat')
        z_q, z_q_mean, z_q_logvar = outputs.get('z_q'), outputs.get('z_q_mean'), outputs.get('z_q_logvar')

        # Reconstruction loss
        # N E_q [ ln p(x|z) ]
        RE = self.robust_loss.lossfun((x-x_hat).reshape(x.shape[0], -1)).mean()

        # Regularization loss
        # N E_q [ ln q(z) - ln p(z) ]
        log_p_z = self.p_z.log_p(z_q)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar)

        KL = (log_q_z - log_p_z).mean()

        # Total negative lower bound loss
        nelbo = RE + kl_weight*KL

        diagnostics = {
            "bpd"   : (nelbo.item()) / (np.prod(x.shape[1:]) * np.log(2.)),
            "nelbo" : nelbo.item(),
            "RE"    : RE.item(),
            "KL"    : KL.item(),
        }
        return nelbo, diagnostics


    def forward(self, x, **kwargs):
        """ Forward pass through the inference and the generative model.
        """
        # z ~ q(z| x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        # x ~ p(x| z)
        x_hat = self.p_x(z_q)
        return {
            "z_q"        : z_q,
            "z_q_mean"   : z_q_mean,
            "z_q_logvar" : z_q_logvar,

            "x_hat"   : x_hat
        }


if __name__ == "__main__":
    pass
