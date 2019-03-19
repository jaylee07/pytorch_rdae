
import os, sys, time, pickle
import numpy as np
import arguments
import dae
import dset
import imgplot
import matplotlib.pyplot as plt

import rdae
import util


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import utils as v_util

def get_sampledict(args):
    if not args.is_partial:
        sample_dict = None
    else:
        sample_dict = dict()
        for i in range(10):
            sample_dict[i] = args.class_ratio[i]

    return sample_dict

def main(args):
    exp_config = util.get_expconfig(args)
    result_dir, save_dir, log_dir = util.get_paths(args, exp_config)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_num}")
    else:
        device = torch.device('cpu')
    if args.data == 'mnist':
        assert args.image_ch == 1
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                        dset.Noise(args)])
        trn_dset = dset.PartialMNIST(root = os.path.join(args.datadir, 'MNIST'),
                                     sample_dict = get_sampledict(args),
                                     train=True,
                                     transform=transform,
                                     target_transform=None)

    if args.model == 'dae':
        model = dae.DAE(args, device, logdir=log_dir)
    elif args.model == 'cdae':
        model = dae.CDAE(args, device, logdir=log_dir)
    elif 'rdae' in args.model or 'rcdae' in args.model:
        model = rdae.RDAE(args, device, logdir=log_dir)

    if args.mode == 'train':
        model.fit(trn_dset)
        if args.save:
            model.save_model(save_dir)

    if args.load:
        assert args.mode == 'infer'
        model.load_model(save_dir)

    loader = torch.utils.data.DataLoader(trn_dset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
    for i, (image, label) in enumerate(loader):
        recon = model.reconstruct(image.to(device))
        break
    tile_images = imgplot.plot_tile_images(image, img_shapes=(28, 28), tile_shapes=(8, 8), tile_spacings=(0, 0))
    tile_recons = imgplot.plot_tile_images(recon, img_shapes=(28, 28), tile_shapes=(8, 8), tile_spacings=(0, 0))
    fig1 = plt.figure()
    plt.imshow(tile_images, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, 'original.jpg'))
    fig2 = plt.figure()
    plt.imshow(tile_recons, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, 'reconstruction.jpg'))

if __name__ == '__main__':
    args = arguments.get_arguments()
    main(args)
