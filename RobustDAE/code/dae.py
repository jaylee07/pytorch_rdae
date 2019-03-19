import os
import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX

from util import *
from networks import *


class DAE(nn.Module):
    def __init__(self, args, device, logdir = None):
        super(DAE, self).__init__()
        self.args = args
        self.device = device
        self.logdir = logdir
        self.encoder = Encoder(args).to(device)
        self.decoder = Decoder(args).to(device)
        self.loss_func = nn.MSELoss(reduction='none')
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.opt = optim.Adam(params=params, lr=args.lr, betas=(0.5, 0.999))

    def make_trnloader(self, dset):
        loader = torch.utils.data.DataLoader(dset, batch_size=self.args.batch_size,
                                             shuffle=True, num_workers=self.args.n_workers, drop_last=True)
        return loader

    def initialize_param(self):
        self.encoder.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def fit(self, trn_dset):
        writer = tensorboardX.SummaryWriter(self.logdir)
        trn_loader = self.make_trnloader(trn_dset)
        self.initialize_param()
        for epoch in range(self.args.inner_epochs):
            self.encoder.train()
            self.decoder.train()
            trn_loss = self.partial_fit(trn_loader)
            if (epoch + 1) % 20 == 0:
                writer.add_scalar('trn_loss', trn_loss, global_step=epoch)
                print(f"In epoch {epoch + 1}, trn_loss = {trn_loss:.4f}")

    def partial_fit(self, trn_loader):
        for idx, (image, label) in enumerate(trn_loader):
            image, label = image.to(self.device), label.to(self.device)
            image_recon = self.reconstruct(image)
            trn_loss = torch.sum(self.loss_func(image_recon, image), dim=(0, 1, 2, 3)) / image.size()[0]
            self.opt.zero_grad()
            trn_loss.backward()
            self.opt.step()
            return trn_loss

    def get_embedding_vector(self, x):
        with torch.no_grad():
            if x.is_cuda and len(self.args.multi_gpus) > 1:
                out = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
            else:
                out = self.encoder(x)
        return out

    def reconstruct(self, x):
        if x.is_cuda and len(self.args.multi_gpus) > 1:
            z = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
            x_recon = nn.parallel.data_parallel(self.decoder, z, device_ids=self.args.multi_gpus)
        else:
            z = self.encoder(x)
            x_recon = self.decoder(z)
        return x_recon

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_dir
        torch.save(self.encoder.state_dict(), os.path.join(save_path, 'encoder.pkl'))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, 'decoder.pkl'))
        print('Save model!')

    def load_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_dir
        self.encoder.load_state_dict(torch.load(os.path.join(save_path, 'encoder.pkl')))
        self.decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.pkl')))


class CDAE(nn.Module):
    def __init__(self, args, device, logdir=None):
        super(CDAE, self).__init__()
        self.args = args
        self.device = device
        self.logdir = logdir
        self.encoder = Encoder_conv2d(args).to(self.device)
        embed_w, embed_h = args.image_size, args.image_size
        for i in range(len(args.kernels)):
            embed_w, embed_h = conv2d_output_size(embed_w, embed_h, args.kernels[i], args.kernels[i], args.strides[i],
                                                  args.paddings[i])
        self.decoder = Decoder_conv2d(args, embed_w, embed_h, out_act_fn=None).to(self.device)
        self.device = device
        self.loss_func = nn.MSELoss(reduction='none')
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.opt = optim.Adam(params=params, lr=0.001, betas=(0.5, 0.999))

    def make_trnloader(self, dset):
        loader = torch.utils.data.DataLoader(dset, batch_size=self.args.batch_size,
                                             shuffle=True, num_workers=self.args.n_workers, drop_last=True)
        return loader

    def initialize_param(self):
        self.encoder.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def fit(self, trn_dset):
        self.initialize_param()
        trn_loader = self.make_trnloader(trn_dset)
        writer = tensorboardX.SummaryWriter(self.logdir)
        for epoch in range(self.args.inner_epochs):
            self.encoder.train()
            self.decoder.train()
            trn_loss = self.partial_fit(trn_loader)
            if (epoch + 1) % 20 == 0:
                writer.add_scalar('trn_loss', trn_loss, global_step=epoch)
                print(f"In epoch {epoch + 1}, trn_loss = {trn_loss:.4f}")

    def partial_fit(self, trn_loader):
        for idx, (image, label) in enumerate(trn_loader):
            image, label = image.to(self.device), label.to(self.device)
            image_recon = self.reconstruct(image)
            trn_loss = torch.sum(self.loss_func(image_recon, image), dim=(0, 1, 2, 3)) / image.size()[0]
            self.opt.zero_grad()
            trn_loss.backward()
            self.opt.step()
            return trn_loss

    def get_embedding_vector(self, x):
        with torch.no_grad():
            if x.is_cuda and len(self.args.multi_gpus) > 1:
                out = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
            else:
                out = self.encoder(x)
        return out

    def reconstruct(self, x):
        if x.is_cuda and len(self.args.multi_gpus) > 1:
            z = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
            x_recon = nn.parallel.data_parallel(self.decoder, z, device_ids=self.args.multi_gpus)
        else:
            z = self.encoder(x)
            x_recon = self.decoder(z)
        return x_recon

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_dir
        torch.save(self.encoder.state_dict(), os.path.join(save_path, 'encoder.pkl'))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, 'decoder.pkl'))
        print('Save model!')

    def load_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_dir
        self.encoder.load_state_dict(torch.load(os.path.join(save_path, 'encoder.pkl')))
        self.decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.pkl')))
#
# class CDAE(nn.Module):
#     def __init__(self, args, device):
#         super(CDAE, self).__init__()
#         self.args = args
#         self.device = device
#         self.encoder = Encoder_conv2d(args).to(self.device)
#         embed_w, embed_h = args.image_size, args.image_size
#         for i in range(len(args.kernels)):
#             embed_w, embed_h = conv2d_output_size(embed_w, embed_h, args.kernels[i], args.kernels[i], args.strides[i],
#                                                   args.paddings[i])
#         self.decoder = Decoder_conv2d(args, embed_w, embed_h, out_act_fn=None).to(self.device)
#         self.encoder.apply(initialize_weights)
#         self.decoder.apply(initialize_weights)
#         self.device = device
#         self.loss_func = nn.MSELoss(reduction='none')
#         params = list(self.encoder.parameters()) + list(self.decoder.parameters())
#         self.opt = optim.Adam(params=params, lr=0.001, betas=(0.5, 0.999))
#
#     def fit(self, trn_loader, corrupt_col_idx, corrupt_row_idx):
#         writer = tensorboardX.SummaryWriter(self.args.log_dir)
#         for epoch in range(self.args.max_epochs):
#             self.encoder.train()
#             self.decoder.train()
#             trn_loss = self.partial_fit(trn_loader, corrupt_col_idx, corrupt_row_idx)
#             if (epoch + 1) % 20 == 0:
#                 writer.add_scalar('trn_loss', trn_loss, global_step=epoch)
#                 print(f"In epoch {epoch + 1}, trn_loss = {trn_loss:.4f}")
#
#     def partial_fit(self, trn_loader, corrupt_col_idx, corrupt_row_idx):
#         for idx, (image, label) in enumerate(trn_loader):
#             image, label = image.to(self.device), label.to(self.device)
#             if self.args.noise_method is not None:
#                 image = add_noise(image, corrupt_col_idx, corrupt_row_idx, self.args.noise_method)
#             image_recon = self.reconstruct(image)
#             trn_loss = torch.sum(self.loss_func(image_recon, image), dim=(0, 1, 2, 3)) / image.size()[0]
#             self.opt.zero_grad()
#             trn_loss.backward()
#             self.opt.step()
#             return trn_loss
#
#     def get_embedding_vector(self, x):
#         if x.is_cuda and len(self.args.multi_gpus) > 1:
#             out = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
#         else:
#             out = self.encoder(x)
#         return out
#
#     def reconstruct(self, x):
#         if x.is_cuda and len(self.args.multi_gpus) > 1:
#             z = nn.parallel.data_parallel(self.encoder, x, device_ids=self.args.multi_gpus)
#             x_recon = nn.parallel.data_parallel(self.decoder, z, device_ids=self.args.multi_gpus)
#         else:
#             z = self.encoder(x)
#             x_recon = self.decoder(z)
#         return x_recon
#
#     def save_model(self, save_path=None):
#         if save_path is None:
#             save_path = self.args.save_dir
#         torch.save(self.encoder.state_dict(), os.path.join(save_path, 'encoder.pkl'))
#         torch.save(self.decoder.state_dict(), os.path.join(save_path, 'decoder.pkl'))
#         print('Save model!')
#
#     def load_model(self, save_path=None):
#         if save_path is None:
#             save_path = self.args.save_dir
#         self.encoder.load_state_dict(torch.load(os.path.join(save_path, 'encoder.pkl')))
#         self.decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.pkl')))
