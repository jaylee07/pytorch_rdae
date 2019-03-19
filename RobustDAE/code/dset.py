from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch

from torchvision import datasets, transforms

class UpdatedDataset(data.Dataset):
    def __init__(self, tensor_x, tensor_y, transform = None):
        self.tensor_x = tensor_x.view(-1, 1, 28, 28)
        self.tensor_y = tensor_y
        self.transform = transform

    def __len__(self):
        return len(self.tensor_x)

    def __getitem__(self, idx):
        return self.tensor_x[idx], self.tensor_y[idx]


class PartialMNIST(data.Dataset):
    def __init__(self, root, sample_dict=None, train=True, transform=None, target_transform=None):
        self.root = os.path.join(root, 'MNIST')
        self.sample_dict = sample_dict
        self.processed_folder = os.path.join(self.root, 'processed')
        self.train = train
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')
        print(self._check_exists())
        if not self._check_exists():
            dset = datasets.MNIST(root=self.root, train=True, download=False)
            del dset
        self.data, self.targets = self._sample()
        self.transform = transform
        self.target_transform = target_transform

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _sample(self):
        if self.train:
            data, targets = torch.load(os.path.join(self.processed_folder, self.training_file))
        else:
            data, targets = torch.load(os.path.join(self.processed_folder, self.test_file))

        if self.sample_dict is not None:
            total_idx = list()
            for i in range(10):
                tmp = ((targets == i).nonzero()).flatten().numpy().tolist()
                class_idx = sorted(np.random.choice(tmp, size=self.sample_dict[i], replace=False).tolist())
                total_idx.extend(class_idx)

            total_idx = sorted(total_idx)
            return data[total_idx], targets[total_idx]

        else:
            return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Noise(object):
    def __init__(self, args):
        self.args = args
        self.corrupt_col_idx = np.random.choice([x for x in range(args.image_size)],
                                                size=args.n_corrupt_cols, replace=False)
        self.corrupt_row_idx = np.random.choice([x for x in range(args.image_size)],
                                                size=args.n_corrupt_rows, replace=False)

    def __call__(self, sample):
        n_sample, n_row, n_col = sample.size()[0], sample.size()[1], sample.size()[2]
        noise = torch.zeros((n_sample, n_row, n_col))
        if self.args.noise_method == 'none':
            pass
        elif self.args.noise_method == 'colwise':
            for col in self.corrupt_col_idx:
                noise[:, :, col] = torch.rand(n_sample, n_row)
        elif self.args.noise_method == 'rowwise':
            for row in self.corrupt_row_idx:
                noise[:, row, :] = torch.rand(n_sample, n_col)
        elif self.args.noise_method == 'rowcol':
            for row in self.corrupt_row_idx:
                for col in self.corrupt_col_idx:
                    noise[:, row, col] = torch.rand(n_sample)
        # elif self.args.noise_method == 'random':
        #     noise = torch.rand(n_sample, n_row, n_col)
        else:
            raise ValueError('Enter the proper noise method')
        sample = sample + noise
        return sample
