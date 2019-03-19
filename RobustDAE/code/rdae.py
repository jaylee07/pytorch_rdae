import os, pickle

import torch
import torch.nn as nn

from dae import DAE, CDAE
from dset import UpdatedDataset, PartialMNIST
from shrink import l1shrink, l21shrink


class RDAE(nn.Module):
    def __init__(self, args, device, logdir = None, tol=1e-7):
        super(RDAE, self).__init__()
        self.args = args
        self.device = device
        self.logdir = logdir
        self.tol = tol
        self.lamb = args.lamb
        if 'rdae' in args.model:
            self.ae = DAE(args, device)
        elif 'rcdae' in args.model:
            self.ae = CDAE(args, device)

    def make_rdaeloader(self, dset):
        loader = torch.utils.data.DataLoader(dset, batch_size=self.args.batch_size,
                                             shuffle=False, num_workers=0, drop_last=False)
        return loader

    def get_flatX(self, loader):
        image_list, label_list = list(), list()
        for _, (image, label) in enumerate(loader):
            image, label = image.to(self.device), label.to(self.device)
            image_list.append(image)
            label_list.append(label)
        flat_images = torch.cat([x.flatten(start_dim=1) for x in image_list], dim=0)
        labels = torch.cat([x for x in label_list])

        return flat_images, labels

    def get_flat_recon_X(self, loader):
        recon_list, label_list = list(), list()
        with torch.no_grad():
            for _, (image, label) in enumerate(loader):
                image, label = image.to(self.device), label.to(self.device)
                recon = self.ae.reconstruct(image)
                recon_list.append(recon)
                label_list.append(label)
        flat_images = torch.cat([x.flatten(start_dim=1) for x in recon_list], dim=0)
        labels = torch.cat([x for x in label_list])

        return flat_images, labels

    def fit(self, trn_dset, verbose=True):
        # Make data loader
        rdae_loader = self.make_rdaeloader(trn_dset)
        # Make flat data
        X, Y = self.get_flatX(rdae_loader)
        # Make L and S
        L = torch.zeros((X.size()[0], X.size()[1])).to(self.device)
        S = torch.zeros((X.size()[0], X.size()[1])).to(self.device)
        mu = (X.size()[0] * X.size()[1]) / torch.norm(X, 1)
        print(f'shrink param: {self.lamb / mu:.4f}')
        LS0 = L + S
        XFnorm = torch.norm(X, 'fro')
        for i in range(self.args.outer_epochs):
            print(f">>{i + 1}th epoch")
            L = X - S
            trn_dset = UpdatedDataset(L.cpu(), Y.cpu())
            print('>>>>start train ae')
            self.ae.fit(trn_dset)
            print('>>>>end train ae')
            print('>>>>Update L, S')
            L, Y = self.get_flat_recon_X(rdae_loader)
            S = X - L
            if 'l1' in self.args.model:
                print('l1 shrink')
                S = l1shrink(self.lamb / mu, S, self.device)
            elif 'l21' in self.args.model:
                print('l21 shrink')
                S = l21shrink(self.lamb / mu, S, self.device)

            c1 = torch.norm(X - L - S, 'fro') / XFnorm
            c2 = mu * torch.norm(LS0 - L - S) / XFnorm

            self.L, self.S = L, S
            if verbose:
                print(f'>>>>c1: {c1:.4f}, c2: {c2:.4f}')
            if c1 < self.tol and c2 < self.tol:
                print('Early break')
                break
            LS0 = L + S

        return self.L, self.S

    def transform(self, x):
        L = x - self.S
        return self.ae.get_embedding_vector(L)

    def reconstruct(self, x):
        return self.ae.reconstruct(x)

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_path
        self.ae.save_model(save_path)
        save_dict = dict()
        save_dict['L'], save_dict['S'] = self.L, self.S
        with open(os.path.join(save_path, 'save_dict.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)

    def load_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_path
        self.ae.load_model(save_path)
        with open(os.path.join(save_path, 'save_dict.pkl'), 'rb') as f:
            save_dict = pickle.load(f)
        self.L, self.S = save_dict['L'], save_dict['S']
