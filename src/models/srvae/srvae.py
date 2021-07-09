from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.utils import args, get_shape
from src.modules import *

from robust_loss_pytorch import AdaptiveLossFunction
from lintegrate import lqag, lqng, lcquad


# ----- NN Model Seleciton -----

if args.model == 'srVAE':
    if args.network == 'densenet32x64':
        from .image_networks.densenet32x64 import *
    else:
        raise NotImplementedError("Model not implemented.")


# ----- Two Staged VAE -----

class srVAE(nn.Module):
    """
    Super-Resolution Variational Auto-Encoder (srVAE).
    A Two Staged Visual Processing Variational AutoEncoder.

    Authors:
    Ioannis Gatopoulos., 2020
    Constantin Stipnieks, 2021
    """
    def __init__(self, x_shape, y_shape=(3, 32, 32), u_dim=args.u_dim, z_dim=args.z_dim, prior=args.prior, device=args.device):
        super().__init__()
        self.device = args.device
        self.x_shape = x_shape
        self.y_shape = (x_shape[0], y_shape[1], y_shape[2])

        self.u_shape = get_shape(u_dim)
        self.z_shape = get_shape(z_dim)

        # q(y|x): deterministic "compressed" transformation
        self.compressed_transform = transforms.Compose([
            transforms.Lambda(lambda X: (X + 1.)/2.),
            transforms.ToPILImage(),
            transforms.Resize((self.y_shape[1], self.y_shape[2])),
            transforms.ToTensor(),
            transforms.Lambda(lambda X: 2*X - 1.)
        ])

        # p(u)
        self.p_u = globals()[prior](self.u_shape)

        # q(u | y)
        self.q_u = q_u(self.u_shape, self.y_shape)

        # p(z | y)
        self.p_z = p_z(self.z_shape, (self.y_shape, self.u_shape))

        # q(z | x)
        self.q_z = q_z(self.z_shape, self.x_shape)

        # p(y | u)
        self.p_y = p_y(self.y_shape, self.u_shape)

        # p(x | y, z)
        self.p_x = p_x(self.x_shape, (self.y_shape, self.z_shape))

        # likelihood distribution
        flat_x_dim = x_shape[0]*x_shape[1]*x_shape[2]
        self.robust_loss_x = AdaptiveLossFunction(num_dims=flat_x_dim, float_dtype=np.float32, device=torch.cuda.current_device())
        flat_y_dim = y_shape[0]*y_shape[1]*y_shape[2]
        self.robust_loss_y = AdaptiveLossFunction(num_dims=flat_y_dim, float_dtype=np.float32, device=torch.cuda.current_device())
        #self.batch_num = batch_num
        #self.consistency_weight = 0

        self.mse_loss = nn.MSELoss()
        #self.recon_loss = partial(dmol_loss)
        #self.sample_distribution = partial(sample_from_dmol)


    def compressed_transformation(self, input):
        y = []
        for x in input:
            y.append(self.compressed_transform(x.cpu()))
        return torch.stack(y).to(self.device)

    '''
    def initialize(self, dataloader):
        """ Data dependent init for weight normalization 
            (Automatically done during the first forward pass).
        """
        with torch.no_grad():
            x, _ = next(iter(dataloader))
            x = x.to(self.device)
            output = self.forward(x)
            self.calculate_elbo(x, output)
        return
    '''

    @staticmethod
    def reparameterize(z_mean, z_log_var):
        """ z ~ N(z| z_mu, z_logvar) """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon


    @torch.no_grad()
    def generate(self, n_samples=20):
        # u ~ p(u)
        u = self.p_u.sample(n_samples=n_samples, device=self.device).to(self.device)

        # p(y|u)
        y_hat = self.p_y(u.contiguous())

        # z ~ p(z|y, u)
        z_p_mean, z_p_logvar = self.p_z((y_hat, u.contiguous()))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_hat = self.p_x((y_hat, z_p))
        return x_hat, y_hat


    @torch.no_grad()
    def reconstruct(self, x, **kwargs):
        outputs = self.forward(x)
        y_hat = outputs.get('y_hat')
        x_hat = outputs.get('x_hat')
        return outputs.get('y'), y_hat, x_hat


    @torch.no_grad()
    def super_resolution(self, y):
        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ p(z|y)
        z_p_mean, z_p_logvar = self.p_z((y, u_q))
        z_p = self.reparameterize(z_p_mean, z_p_logvar)

        # x ~ p(x|y,z)
        x_hat = self.p_x((y, z_p))
        return x_hat


    def update_consistency_weight(self):
        # linearly icrease weight over 10 epochs until it reaches 5
        step_size = 5 / (self.batch_num*10)
        if self.consistency_weight <= 5 - step_size:
            self.consistency_weight += step_size


    def discrete_recon_loss(self, loss_fn, gt, pred):
        '''
        Approximation of discretized adaptive robust loss function through trapezoidal rule
        '''

        bin_size = 1/127.5
        sample_num = 100
        step_size = bin_size/sample_num
        sample = gt - bin_size/2 + step_size/2
        bin_prob_normal = 0.

        # normal case
        for i in range(sample_num):
            bin_prob_normal += (-loss_fn((pred-sample).reshape(gt.shape[0], -1))).exp() * step_size
            sample += step_size

        # lower edge case -1
        mask_lower = gt.reshape(gt.shape[0], -1)==-1.
        bin_size = 2 + 1/127.5
        sample_num = 100
        step_size_edge = bin_size/sample_num
        sample = -3 + step_size_edge/2
        bin_prob_low_edge = 0.
        for i in range(sample_num):
            bin_prob_low_edge += (-loss_fn((pred-sample).reshape(gt.shape[0], -1))).exp() * step_size_edge
            sample += step_size_edge

        # upper edge case 1
        mask_upper = gt.reshape(gt.shape[0], -1)==1.
        bin_size = 2 + 1/127.5
        sample_num = 100
        sample = gt - 1/127.5 + step_size_edge/2
        bin_prob_high_edge = 0.
        for i in range(sample_num):
            bin_prob_high_edge += (-loss_fn((pred-sample).reshape(gt.shape[0], -1))).exp() * step_size_edge
            sample += step_size_edge

        final_discrete_prob = bin_prob_normal*mask_lower.logical_not() + bin_prob_low_edge*mask_lower
        final_discrete_prob = bin_prob_normal*mask_upper.logical_not() + bin_prob_high_edge*mask_upper

        return torch.log(final_discrete_prob)


    def calculate_elbo(self, x, outputs, kl_weight=1, **kwargs):
        # unpack variables
        y, x_hat, y_hat = outputs.get('y'), outputs.get('x_hat'), outputs.get('y_hat')
        u_q, u_q_mean, u_q_logvar = outputs.get('u_q'), outputs.get('u_q_mean'), outputs.get('u_q_logvar')
        z_q, z_q_mean, z_q_logvar = outputs.get('z_q'), outputs.get('z_q_mean'), outputs.get('z_q_logvar')
        z_p_mean, z_p_logvar = outputs.get('z_p_mean'), outputs.get('z_p_logvar')

        # When calculating the bits per dimension, we need to use the discretized likelihood
        RE_x = -self.discrete_recon_loss(self.robust_loss_x.lossfun, x, x_hat).sum(1).mean()
        RE_y = -self.discrete_recon_loss(self.robust_loss_y.lossfun, y, y_hat).sum(1).mean()

        #RE_x = self.robust_loss_x.lossfun((x-x_hat).reshape(x.shape[0], -1)).mean()
        #RE_y = self.robust_loss_y.lossfun((y-y_hat).reshape(x.shape[0], -1)).mean()

        # Regularization loss
        log_p_u = self.p_u.log_p(u_q, dim=1)
        log_q_u = log_normal_diag(u_q, u_q_mean, u_q_logvar)
        KL_u = (log_q_u - log_p_u).mean()

        log_p_z = log_normal_diag(z_q, z_p_mean, z_p_logvar)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar)
        KL_z = (log_q_z - log_p_z).mean()

        consistency_loss = self.mse_loss(y_hat, F.interpolate(x_hat, size=[self.y_shape[1], self.y_shape[2]], align_corners=False, mode='bilinear'))
        # Total lower bound loss
        nelbo = RE_x + RE_y + 1*KL_u + 1*KL_z# + 5*consistency_loss

        diagnostics = {
            "bpd"   : (nelbo.item()) / (np.prod(x.shape[1:]) * np.log(2.)),
            "nelbo" : nelbo.item(),

            "RE"    : (RE_x + RE_y).mean().item(),
            "RE_x"  : RE_x.mean().item(),
            "RE_y"  : RE_y.mean().item(),

            "KL"    : (KL_z + KL_u).mean().item(),
            "KL_u"  : KL_u.mean().item(),
            "KL_z"  : KL_z.mean().item(),
        }
        return nelbo, diagnostics


    def forward(self, x, **kwargs):
        """ Forward pass through the inference and the generative model. """
        # y ~ f(x) (deterministic)
        y = self.compressed_transformation(x)

        # u ~ q(u| y)
        u_q_mean, u_q_logvar = self.q_u(y)
        u_q = self.reparameterize(u_q_mean, u_q_logvar)

        # z ~ q(z| x, y)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # y ~ p(y| u)
        y_hat = self.p_y(u_q)

        # x ~ p(x| y, z)
        x_hat = self.p_x((y_hat, z_q))

        # z ~ p(z| x)
        z_p_mean, z_p_logvar = self.p_z((y_hat, u_q))

        return {
            'u_q_mean'   : u_q_mean,
            'u_q_logvar' : u_q_logvar,
            'u_q'        : u_q,

            'z_q_mean'   : z_q_mean,
            'z_q_logvar' : z_q_logvar,
            'z_q'        : z_q,

            'z_p_mean'   : z_p_mean,
            'z_p_logvar' : z_p_logvar,

            'y'          : y,
            'y_hat'   : y_hat,

            'x_hat'   : x_hat
        }


if __name__ == "__main__":
    pass
