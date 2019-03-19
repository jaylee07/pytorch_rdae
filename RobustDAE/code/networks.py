import torch
import torch.nn as nn
from collections import OrderedDict
from util import *


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        # m.weight.data.normal_(1.0, 0.02)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        if args.act_fn == 'relu':
            act_fn = nn.ReLU()
        assert len(args.dims) >= 2
        layers = OrderedDict()
        for i in range(1, len(args.dims)):
            in_dim, out_dim = args.dims[i - 1], args.dims[i]
            layers[f'layer{i}'] = nn.Linear(in_dim, out_dim)
            layers[f'act_fn{i}'] = act_fn
        del layers[f'act_fn{i}']
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = x.view(-1, self.args.image_ch * self.args.image_size * self.args.image_size)
        out = self.layers(x)
        return out


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        if args.act_fn == 'relu':
            act_fn = nn.ReLU()
        assert len(args.dims) >= 2
        layers = OrderedDict()
        dims = [x for x in reversed(args.dims)]
        for i in range(1, len(args.dims)):
            in_dim, out_dim = dims[i - 1], dims[i]
            layers[f'layer{i}'] = nn.Linear(in_dim, out_dim)
            layers[f'act_fn{i}'] = act_fn
        del layers[f'act_fn{i}']
        if args.out_act_fn == 'none':
            self.out_act_fn = None
        elif args.out_act_fn == 'sigmoid':
            self.out_act_fn = nn.Sigmoid()
        elif args.out_act_fn == 'tanh':
            self.out_act_fn = nn.Tanh()
        layers[f'act_fn{i}'] = self.out_act_fn
        self.layers = nn.Sequential(layers)

    def forward(self, z):
        out = self.layers(z)
        out = out.view(-1, self.args.image_ch, self.args.image_size, self.args.image_size)
        return out


class Encoder_conv2d(nn.Module):
    def __init__(self, args):
        super(Encoder_conv2d, self).__init__()
        self.args = args
        self.use_fc = args.use_fc
        if args.act_fn == 'relu':
            act_fn = nn.ReLU()
        k = args.kernels
        s = args.strides
        p = args.paddings
        layers = OrderedDict()
        w, h = args.image_size, args.image_size
        in_ch, out_ch = args.image_ch, args.n_ch
        for i in range(len(k) - 1):
            layers[f'conv{i + 1}'] = nn.Conv2d(in_ch, out_ch, k[i], s[i], p[i], bias=True)
            layers[f'bn{i + 1}'] = nn.BatchNorm2d(num_features=out_ch)
            layers[f'act_fn{i + 1}'] = act_fn
            w, h = conv2d_output_size(w, h, k[i], k[i], s[i], p[i])
            in_ch, out_ch = out_ch, out_ch * 2
        i = i + 1
        layers[f'conv{i + 1}'] = nn.Conv2d(in_ch, out_ch, k[i], s[i], p[i], bias=True)
        self.layers = nn.Sequential(layers)
        w, h = conv2d_output_size(w, h, k[i], k[i], s[i], p[i])
        if self.use_fc:
            self.fc = nn.Linear(w * h * out_ch, args.embed_dim)

    def forward(self, x):
        out = self.layers(x)
        if self.use_fc:
            out = torch.flatten(out, start_dim=1)
            out = self.fc(out)
        return out


class Decoder_conv2d(nn.Module):
    def __init__(self, args, embed_w, embed_h, out_act_fn=None, use_fc=False):
        super(Decoder_conv2d, self).__init__()
        self.args = args
        self.use_fc = args.use_fc
        self.embed_w = embed_w
        self.embed_h = embed_h
        if args.act_fn == 'relu':
            act_fn = nn.ReLU()
        k = [x for x in reversed(args.kernels)]
        s = [x for x in reversed(args.strides)]
        p = [x for x in reversed(args.paddings)]
        in_ch = args.n_ch * (2 ** (len(k) - 1))
        out_ch = int(in_ch / 2)
        if self.use_fc:
            self.fc = nn.Linear(args.embed_dim, embed_w * embed_h * in_ch)
        size_w, size_h = embed_w, embed_h

        layers = OrderedDict()
        for i in range(len(k) - 1):
            layers[f'convtr{i + 1}'] = nn.ConvTranspose2d(in_ch, out_ch, k[i], s[i], p[i], bias=True)
            layers[f'bn{i + 1}'] = nn.BatchNorm2d(num_features=out_ch)
            layers[f'act{i + 1}'] = act_fn
            size_w, size_h = convtr2d_output_size(size_w, size_h, k[i], k[i], s[i], p[i])
            in_ch, out_ch = out_ch, int(out_ch / 2)
        i = i + 1
        layers[f'convtr{i + 1}'] = nn.ConvTranspose2d(in_ch, args.image_ch, k[i], s[i], p[i], bias=True)
        size_w, size_h = convtr2d_output_size(size_w, size_h, k[i], k[i], s[i], p[i])
        self.layers = nn.Sequential(layers)
        if args.out_act_fn == 'none':
            self.out_act_fn = None
        elif args.out_act_fn == 'sigmoid':
            self.out_act_fn = nn.Sigmoid()
        elif args.out_act_fn == 'tanh':
            self.out_act_fn = nn.Tanh()

    def forward(self, z):
        if self.use_fc:
            z = self.fc(z)
            out_ch = self.args.n_ch * (2 ** (len(self.args.kernels) - 1))
            z = z.view(-1, out_ch, self.embed_h, self.embed_w)
        out = self.layers(z)
        if self.out_act_fn is not None:
            out = self.out_act_fn(out)
        return out
