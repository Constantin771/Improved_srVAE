import random
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from .args import args


TF_NORMALIZE  = True
JPG_NORMALIZE = True


# ----- Generate -----

def generate(model, n_samples, epoch=0, writer=None):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'Generate data...'+30*' '), end='\r')
    
    models_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    n_rows = int(n_samples**0.5)

    generator = model.module.generate if isinstance(model, nn.DataParallel) else model.generate

    if models_name == 'VAE':
        x = generator(n_samples)
        if writer:
            writer.add_image('generation/x', make_grid(x, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            fname = './images/' + 'generated_images_' + str(epoch) + '.jpg'
            save_image(make_grid(x, nrow=n_rows, normalize=JPG_NORMALIZE), fname)

    elif models_name == 'srVAE':
        x, y = generator(n_samples)
        if writer:
            writer.add_image('generation/x', make_grid(x, nrow=7, normalize=TF_NORMALIZE), epoch)
            writer.add_image('generation/y', make_grid(y, nrow=7, normalize=TF_NORMALIZE), epoch)
        else:
            fname_x = './images/' + 'generated_images_x_' + str(epoch) + '.jpg'
            fname_y = './images/' + 'generated_images_y_' + str(epoch) + '.jpg'
            save_image(make_grid(x, nrow=7, normalize=JPG_NORMALIZE), fname_x)
            save_image(make_grid(y, nrow=7, normalize=JPG_NORMALIZE), fname_y)
            save_image(make_grid(nn.functional.interpolate(y, size=x.shape[2:]), nrow=n_rows, normalize=JPG_NORMALIZE), './images/' + 're_generated_images_y_' + str(epoch) + '.jpg')
    else:
        pass

    return


def generate_latent_interpol(model, n_samples=100, epoch=0, writer=None):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'Generate data...'+30*' '), end='\r')
    
    models_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    n_rows = int(n_samples**0.5)

    generator = model.module.generate_latent_var if isinstance(model, nn.DataParallel) else model.generate_latent_interpol

    if models_name == 'VAE':
        x = generator()
        if writer:
            writer.add_image('generation/x', make_grid(x, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            fname = './images/' + 'generated_images_' + str(epoch) + '.jpg'
            save_image(make_grid(x, nrow=n_rows, normalize=JPG_NORMALIZE), fname)

    elif models_name == 'srVAE':
        x, y = generator()
        if writer:
            writer.add_image('generation/x', make_grid(x, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
            writer.add_image('generation/y', make_grid(y, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            fname_x = './images/' + 'generated_images_x_' + str(epoch) + '.jpg'
            fname_y = './images/' + 'generated_images_y_' + str(epoch) + '.jpg'
            save_image(make_grid(x, nrow=n_rows, normalize=JPG_NORMALIZE), fname_x)
            save_image(make_grid(y, nrow=n_rows, normalize=JPG_NORMALIZE), fname_y)
            save_image(make_grid(nn.functional.interpolate(y, size=x.shape[2:]), nrow=n_rows, normalize=JPG_NORMALIZE), './images/' + 're_generated_images_y_' + str(epoch) + '.jpg')
    else:
        pass

    return


# ----- Reconstruct -----

def reconstruction(model, dataloader, n_samples, epoch=0, writer=None):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'Reconstruct data...'+30*' '), end='\r')

    n_samples = min(args.batch_size, n_samples)
    n_rows = int(n_samples**0.5)

    models_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    reconstructor = model.module.reconstruct if isinstance(model, nn.DataParallel) else model.reconstruct

    # sample a random batch
    batch = random.randint(0, len(dataloader))
    for i, (x, labels) in enumerate(dataloader):
        x, labels = x[:n_samples].to(args.device), labels[:n_samples].to(args.device)
        if i==batch-1:
            break

    if models_name=='VAE':
        x_hat = reconstructor(x)
        if writer:
            writer.add_image('reconstruction/x', make_grid(x, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
            writer.add_image('reconstruction/x_rec', make_grid(x_hat, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            fname = './images/' + 'reconstructions_' + str(epoch) + '.jpg'
            imgs = torch.cat((x, x_hat), 0)
            save_image(make_grid(imgs, nrow=n_samples, normalize=JPG_NORMALIZE), fname)

    elif models_name=='srVAE':
        # reconstruction
        y, y_hat, x_hat = reconstructor(x)
        if writer:
            writer.add_image('reconstruction/x', make_grid(x, nrow=7, normalize=TF_NORMALIZE), epoch)
            writer.add_image('reconstruction/y', make_grid(y, nrow=7, normalize=TF_NORMALIZE), epoch)
            writer.add_image('reconstruction/x_rec', make_grid(x_hat, nrow=7, normalize=TF_NORMALIZE), epoch)
            writer.add_image('reconstruction/y_rec', make_grid(y_hat, nrow=7, normalize=TF_NORMALIZE), epoch)
        else:
            fname_x = './images/' + 'reconstructed_images_x_' + str(epoch) + '.jpg'
            fname_y = './images/' + 'reconstructed_images_y_' + str(epoch) + '.jpg'
            joint_x = torch.cat((x, x_hat), 0)
            joint_y = torch.cat((y, y_hat), 0)
            save_image(make_grid(joint_x, nrow=7, normalize=JPG_NORMALIZE), fname_x)
            save_image(make_grid(joint_y, nrow=7, normalize=JPG_NORMALIZE), fname_y)

        # super-resolution
        super_resolution = model.module.super_resolution if isinstance(model, nn.DataParallel) else model.super_resolution
        x_super = super_resolution(y)
        if writer:
            writer.add_image('super_resolution/x', make_grid(x, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
            writer.add_image('super_resolution/y', make_grid(y, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
            writer.add_image('super_resolution/x_super', make_grid(x_super, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            fname = './images/' + 'super_res_' + str(epoch) + '.jpg'
            y_img = nn.functional.interpolate(y, size=x.shape[2:])
            images = torch.cat((y_img, x, x_super), 0)
            save_image(make_grid(images, nrow=n_samples, normalize=JPG_NORMALIZE), fname)
    else:
        pass

    return


# ----- Image Interpolation -----

def interpolation(model, dataloader, n_samples, epoch=0, writer=None):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'Image interpolation...'+24*' '), end='\r')

    n_samples += 2
    imgs, _ = next(iter(dataloader))
    #idx1, idx2 = random.randint(0, imgs.shape[0]), random.randint(0, imgs.shape[0])
    #img_1, img_2 = imgs[idx1].to(args.device).unsqueeze(0), imgs[idx2].to(args.device).unsqueeze(0)
    indices = np.random.choice(imgs.shape[0], 20, replace=False)
    imgs = imgs[indices].to(args.device)

    models_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__

    if models_name in ['VAE']:
        encoder = model.module.q_z if isinstance(model, nn.DataParallel) else model.q_z
        decoder = model.module.p_x if isinstance(model, nn.DataParallel) else model.p_x
        reparameterize = model.module.reparameterize if isinstance(model, nn.DataParallel) else model.reparameterize
        sample_distribution = model.module.sample_distribution if isinstance(model, nn.DataParallel) else model.sample_distribution

        # get latent representations
        z1_mu, z1_logvar = encoder(img_1)
        z2_mu, z2_logvar = encoder(img_2)
        z1 = reparameterize(z1_mu, z1_logvar)
        z2 = reparameterize(z2_mu, z2_logvar)

        # Initialize the interpolation space
        interpolation_space = np.linspace(z1.cpu().detach().numpy(), z2.cpu().detach().numpy(), n_samples)
        code_list = []
        for code in interpolation_space:
            z = torch.from_numpy(code).float().to(args.device) * torch.ones(*z1.shape).to(args.device)
            code_list.append(z)

        z = torch.stack(code_list, dim=0).squeeze(1)

        # generate
        x_logits = decoder(z)
        sample_distribution = model.module.sample_distribution if isinstance(model, nn.DataParallel) else model.sample_distribution
        x_hat = sample_distribution(x_logits)

        if writer:
            writer.add_image('image_completion/x_rec', make_grid(x_hat, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
            writer.add_image('image_completion/x', make_grid(x_img, nrow=n_rows, normalize=TF_NORMALIZE), epoch)
        else:
            save_image(make_grid(x_hat, nrow=n_samples, normalize=JPG_NORMALIZE), 'images/image_interpolation.jpg')

    elif models_name in ['srVAE']:
        q_u = model.module.q_u if isinstance(model, nn.DataParallel) else model.q_u
        p_y = model.module.p_y if isinstance(model, nn.DataParallel) else model.p_y
        p_z = model.module.p_z if isinstance(model, nn.DataParallel) else model.p_z
        p_x = model.module.p_x if isinstance(model, nn.DataParallel) else model.p_x
        reparameterize = model.module.reparameterize if isinstance(model, nn.DataParallel) else model.reparameterize

        y_img_1, y_img_2 = model.module.compressed_transformation(imgs[:10]), model.module.compressed_transformation(imgs[10:])

        # get latent representations
        u1_mu, u1_logvar = q_u(y_img_1)
        u2_mu, u2_logvar = q_u(y_img_2)
        u1 = reparameterize(u1_mu, u1_logvar)
        u2 = reparameterize(u2_mu, u2_logvar)

        # Initialize the interpolation space
        interpolation_space = np.linspace(u1.cpu().detach().numpy(), u2.cpu().detach().numpy(), n_samples)
        code_list = []
        for code in interpolation_space:
            u = torch.from_numpy(code).float().to(args.device) * torch.ones(*u1.shape).to(args.device)
            code_list.append(u)

        u = torch.stack(code_list, dim=0).squeeze(1)
        u = u.transpose(0, 1).reshape(-1, u.shape[2], u.shape[3], u.shape[4])

        # generate
        y_hat = p_y(u)
        z_p_mean, z_p_logvar = p_z((y_hat, u))
        z_p = reparameterize(z_p_mean, z_p_logvar)

        x_hat = p_x((y_hat, z_p))
        x_hat = x_hat.reshape(10, n_samples, 3, 64, 64).transpose(0, 1)
        x_hat = torch.cat((imgs[:10].unsqueeze(0), x_hat, imgs[10:].unsqueeze(0)))
        x_hat = x_hat.transpose(0, 1).reshape(-1, 3, 64, 64)

        if writer:
            writer.add_image('image_completion/x_rec', make_grid(x_hat, nrow=n_samples+2, normalize=TF_NORMALIZE), epoch)
        else:
            save_image(make_grid(x_hat, nrow=n_samples, normalize=JPG_NORMALIZE), 'images/image_interpolation.jpg')
    else:
        pass

    return


if __name__ == "__main__":
    pass
