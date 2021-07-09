import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from rational.torch import Rational

from src.utils import args



# ----- Helpers -----

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        if isinstance(unflatten_size, tuple):
            self.c = unflatten_size[0]
            self.h = unflatten_size[1]
            self.w = unflatten_size[2]
        elif isinstance(unflatten_size, int):
            self.c = unflatten_size
            self.h = 1
            self.w = 1

    def forward(self, x):
        return x.view(x.size(0), self.c, self.h, self.w)



class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


# Dict of different skip connection types
skip_connections = dict([
    ('downsample', lambda C, drop_prob: nn.Sequential(Interpolate(scale=.5), Conv2d(C, int(C * 2), kernel_size=1, drop_prob=drop_prob, act=None))),
    ('upsample', lambda C, drop_prob: nn.Sequential(Interpolate(scale=2), Conv2d(C, int(C / 2), kernel_size=1, drop_prob=drop_prob, act=None))),
    ('multiply', lambda C, drop_prob: Conv2d(C, int(C * 2), kernel_size=1, drop_prob=drop_prob, act=None)),
    ('identity', lambda C, drop_prob: Identity())
])

# Dict of different final layers in residual groups
final_layers = dict([
    ('downsample', lambda C, drop_prob: Downsample(C, 2*C, drop_prob=drop_prob)),
    ('upsample', lambda C, drop_prob: Upsample(C, C//2, drop_prob=drop_prob)),
    ('multiply', lambda C, drop_prob: Conv2d(C, 2*C, kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None)),
    ('identity', lambda C, drop_prob: Identity())
])


# ----- 2D Convolutions -----

# Conv2d init_parameters from: https://github.com/vlievin/biva-pytorch/blob/master/biva/layers/convolution.py
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, weightnorm=False, act=True, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm
        self.initialized = True

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        #if act is not None:
        #   print(act)
        self.act = act() if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.initialized = False
            self.conv = nn.utils.weight_norm(self.conv, dim=0, name="weight")

    def forward(self, input):
        if not self.initialized:
            self.init_parameters(input)
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        self.initialized = True
        if self.weightnorm:
            # initial values
            self.conv._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.conv._parameters['weight_g'].data.fill_(1.)
            self.conv._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.conv(x)
            t = x.view(x.size()[0], x.size()[1], -1)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(-1, t.size()[-1])
            m_init, v_init = torch.mean(t, 0), torch.var(t, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)

            self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[:, None].view(
                self.conv._parameters['weight_g'].data.size())
            self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            return scale_init[None, :, None, None] * (x - m_init[None, :, None, None]) 


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, output_padding=0, dilation=1, groups=1, bias=True, weightnorm=False, act=True, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm
        self.initialized = True

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding,
                                       dilation=dilation, groups=groups, bias=bias)
        #if act is not None:
        #    print(act)
        self.act = act() if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.initialized = False
            self.conv = nn.utils.weight_norm(self.conv, dim=1, name="weight")

    def forward(self, input):
        if not self.initialized:
            print("PROBLEM")
            self.init_parameters(input)
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        self.initialized = True
        if self.weightnorm:
            # initial values
            self.conv._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.conv._parameters['weight_g'].data.fill_(1.)
            self.conv._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.conv(x)
            t = x.view(x.size()[0], x.size()[1], -1)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(-1, t.size()[-1])
            m_init, v_init = torch.mean(t, 0), torch.var(t, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)

            self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[None,:].view(
                self.conv._parameters['weight_g'].data.size())
            self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            return scale_init[None, :, None, None] * (x - m_init[None, :, None, None])


# ----- Up and Down Sampling -----

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            Conv2d(in_channels, out_channels,
                   kernel_size=3, stride=2, padding=1, drop_prob=drop_prob, act=None)
        )

    def forward(self, input):
        return self.core_nn(input)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels,
                            kernel_size=3, stride=2, padding=1, 
                            output_padding=1, drop_prob=drop_prob, act=None)
        )

    def forward(self, input):
        x, latent = input['x'], input['z']
        y = self.core_nn(x)

        return {'x': y, 'z': latent}


# ----- Gated/Attention Blocks -----

class CALayer(nn.Module):
    """
    ChannelWise Gated Layer.
    """
    def __init__(self, channel, act, reduction=4, drop_prob=0.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if channel < reduction:
            reduction = 1
        self.ca_block = nn.Sequential(
            Conv2d(channel, channel // reduction,
                   kernel_size=1, stride=1, padding=0, act=act, drop_prob=drop_prob),
            Conv2d(channel // reduction, channel,
                   kernel_size=1, stride=1, padding=0, act=None, drop_prob=drop_prob),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca_block(y)

        return x * y


class Adaptive_ca_layer(nn.Module):
    def __init__(self, channel, latent_dim, drop_prob=0.0):
        super(Adaptive_ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channel + latent_dim, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.drop_prob = drop_prob

    def forward(self, x, latent):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = F.dropout(self.linear(torch.cat((y.squeeze(), latent.flatten(start_dim=1)), -1)), p=self.drop_prob, training=True)

        y = self.sigmoid(y)

        return x * y.unsqueeze(-1).unsqueeze(-1).expand_as(x)


class CA_Block(nn.Module):
    def __init__(self, in_channels, drop_prob=0.0, act=Rational):
        super().__init__()
        self.dense_block = nn.Sequential(
            Conv2d(in_channels, in_channels,
                   kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=None),
            nn.BatchNorm2d(in_channels),
            act(),
            Conv2d(in_channels, in_channels,
                   kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None),
            nn.BatchNorm2d(in_channels),
            CALayer(in_channels, act=act, drop_prob=drop_prob)
        )


    def forward(self, input):
        y = self.dense_block(input)

        return input + y


class Residual_CA_Group(nn.Module):
    def __init__(self, in_channels, steps, type, drop_prob=0.0, batch_norm=True, act=Rational):
        super().__init__()

        self.skip = skip_connections[type](in_channels, drop_prob=drop_prob)
        net = []
        for step in range(steps):
            net.append(CA_Block(in_channels, drop_prob=drop_prob, act=act))
        net.append(final_layers[type](in_channels, drop_prob=drop_prob))

        self.core_nn = nn.Sequential(*net)


    def forward(self, input):
        skip = self.skip(input)
        return skip + self.core_nn(input)


class Adaptive_CA_Block(nn.Module):
    def __init__(self, in_channels, sa_kernel_size, drop_prob=0.0, batch_norm=True, act=Rational):
        super().__init__()
        self.dense_block = nn.Sequential(
            Conv2d(in_channels, in_channels,
                   kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=None),
            nn.BatchNorm2d(in_channels),
            act(),
            Conv2d(in_channels, in_channels,
                   kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None),
            nn.BatchNorm2d(in_channels)
        )
        self.ca_layer = Adaptive_ca_layer(in_channels, args.z_dim, drop_prob=drop_prob)


    def forward(self, input):
        x, latent = input['x'], input['z']
        
        y = self.dense_block(x)
        y = self.ca_layer(y, latent)

        return {'x': x + y, 'z': latent}


class Residual_AdaCA_Group(nn.Module):
    def __init__(self, in_channels, steps, sa_kernel_size=7, drop_prob=0.0, batch_norm=True, act=Rational):
        super().__init__()

        self.skip = skip_connections['upsample'](in_channels, drop_prob=drop_prob)
        net = []
        for step in range(steps):
            net.append(Adaptive_CA_Block(in_channels, sa_kernel_size, drop_prob=drop_prob, batch_norm=batch_norm, act=act))
        net.append(Upsample(in_channels, in_channels//2, drop_prob=drop_prob))
        self.core_nn = nn.Sequential(*net)

    def forward(self, input):
        x, latent = input['x'], input['z']

        skip = self.skip(x)
        return {'x': skip + self.core_nn(input)['x'], 'z': latent}


class Network(nn.Module):
    def __init__(self, in_channels, out_channels, steps, blocks, drop_prob=0.0, out_act=Rational):
        super().__init__()

        skip = []
        inplanes = in_channels
        for i in range(blocks):
            skip.append(skip_connections['multiply'](inplanes, drop_prob=drop_prob))
            inplanes *= 2
        self.skip_nn = nn.Sequential(*skip)

        # downscale block
        net = []
        for i in range(blocks):
            net.append(Residual_CA_Group(in_channels, steps, 'multiply', drop_prob=drop_prob, act=Rational))
            in_channels *= 2
        self.core_nn = nn.Sequential(*net)

        # output layer
        self.out_nn = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=out_act),
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=None))


    def forward(self, input):
        skip = self.skip_nn(input)
        return self.out_nn(skip + self.core_nn(input))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, steps, scale_factor, drop_prob=0.0):
        super().__init__()

        skip = []
        inplanes = 32
        for i in range(scale_factor):
            skip.append(skip_connections['downsample'](inplanes, drop_prob=drop_prob))
            inplanes *= 2
        self.skip_nn = nn.Sequential(*skip)

        self.input_nn = nn.Sequential(
            Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, act=Rational),
            Conv2d(8, 32, kernel_size=3, stride=1, padding=1, act=None))
        in_channels = 32

        # downscale block
        net = []
        for i in range(scale_factor):
            net.append(Residual_CA_Group(in_channels, steps, 'downsample', drop_prob=drop_prob, act=Rational))
            in_channels *= 2

        # output block
        net.append(Residual_CA_Group(in_channels, steps, 'identity', drop_prob=drop_prob, act=Rational))
        self.core_nn = nn.Sequential(*net)

        # output layer
        self.out_nn = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=Rational),
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=None))


    def forward(self, input):
        y = self.input_nn(input)
        skip = self.skip_nn(y)
        y = self.core_nn(y)
        return self.out_nn(skip + y)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, steps=3, scale_factor=2, drop_prob=0.0, out_act=nn.Tanh):
        super().__init__()

        skip = []
        inplanes = 256
        for i in range(scale_factor):
            skip.append(skip_connections['upsample'](inplanes, drop_prob=drop_prob))
            inplanes = inplanes//2
        self.skip_nn = nn.Sequential(*skip)

        self.input_nn = nn.Sequential(
            Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=Rational),
            Conv2d(16, 256, kernel_size=1, stride=1, padding=0, drop_prob=drop_prob, act=None))

        # upsample block
        net = []
        in_channels = 256
        for i in range(scale_factor):
            net.append(Residual_AdaCA_Group(in_channels, steps, drop_prob=drop_prob, act=Rational))
            in_channels = in_channels//2

        self.core_nn = nn.Sequential(*net)

        # output block
        self.out_nn = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=out_act),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None))


    def forward(self, input):
        y = self.input_nn(input)
        skip = self.skip_nn(y)
        y = self.core_nn({'x': y, 'z': input})['x']
        return self.out_nn(skip + y)


if __name__ == "__main__":
    pass
