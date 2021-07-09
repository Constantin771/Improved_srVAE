""" PyTorch implementation of VAE and Super-Resolution VAE.

    Repository Authors:
        Codebase adapted from Ioannis Gatopoulos, 2020
        Constantin Stipnieks, 2021
"""
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import *
import numpy as np
import random
from torchvision import transforms
from rtpt import RTPT


def train_model(dataset, model, time, writer=None):
    train_loader, valid_loader, test_loader = dataloader(dataset)
    data_shape = get_data_shape(train_loader)

    model = nn.DataParallel(globals()[model](data_shape).to(args.device))

    criterion = ELBOLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)
    scheduler = LowerBoundedExponentialLR(optimizer, gamma=0.999999, lower_bound=0.0001)

    n_parameters(model, writer)

    rtpt = RTPT(name_initials='CS', experiment_name='srVAE', max_iterations=args.epochs)
    rtpt.start()

    for epoch in range(0, args.epochs):
        rtpt.step()
        # Train and Validation epoch
        train_losses = train(model, criterion, optimizer, scheduler, train_loader, epoch)
        valid_losses = evaluate(model, criterion, valid_loader)
        # Visual Evaluation
        generate(model, args.n_samples, epoch, writer)
        reconstruction(model, valid_loader, args.n_samples, epoch, writer)
        # Saving Model and Logging
        is_saved = save_model(model, optimizer, valid_losses['nelbo'], epoch, time)
        logging(epoch, train_losses, valid_losses, writer)


def load_and_evaluate(dataset, model, writer=None):
    pth = './src/models/'

    # configure paths
    pth = os.path.join(pth, 'pretrained', args.model, args.dataset)
    pth_train = os.path.join(pth, 'trainable', '05-06-2021', '00-26-51', 'Model_minValLoss.pth')

    # get data
    train_loader, valid_loader, test_loader = dataloader(dataset)
    data_shape = get_data_shape(train_loader)
    next(iter(train_loader))

    # define model
    model = globals()[model](data_shape).to(args.device)

    # load trained weights for inference
    checkpoint = torch.load(pth_train)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model successfully loaded!')
    except RuntimeError:
        print('* Failed to load the model. Parameter mismatch.')
        quit()
    model = nn.DataParallel(model).to(args.device)
    model.eval()
    criterion = ELBOLoss()
    # Evaluation of the model
    # --- calculate elbo ---
    #test_losses = evaluate(model, criterion, test_loader)
    #print('ELBO: {} bpd'.format(test_losses['bpd']))

    # --- image generation ---
    generate(model, n_samples=42, writer=writer)

    # --- image reconstruction ---
    reconstruction(model, valid_loader, n_samples=42, writer=writer)

    # --- image interpolation ---
    interpolation(model, test_loader, n_samples=5, writer=writer)

    # --- calculate nll ---
    #bpd = calculate_nll(model, test_loader, criterion, args, iw_samples=args.iw_test)
    #print('NLL with {} weighted samples: {:4.2f}'.format(args.iw_test, bpd))



# ----- main -----

def main():
    # Print configs
    print_args(args)

    # Control random seeds
    fix_random_seed(seed=args.seed)
    # Initialize TensorBoad writer (if enabled)
    writer = None
    time = datetime.now().strftime("/%d-%m-%Y/%H-%M-%S")
    if args.use_tb:
        writer = SummaryWriter(log_dir='./logs/'+args.dataset+'_'+args.model+'_'+args.tags +
                               time +"srVAE")
        writer.add_text('args', namespace2markdown(args))

    # Train model
    train_model(args.dataset, args.model, time, writer)

    # Evaluate best (latest saved) model
    #load_and_evaluate(args.dataset, args.model, writer)

    # End Experiment
    if args.use_tb:
        writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'=')


# ----- python main.py -----

if __name__ == "__main__":
    main()
