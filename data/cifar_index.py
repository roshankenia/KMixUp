from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, multiclass_noisify
import sys
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class CIFAR10Index(data.Dataset):

    def __init__(self, train_data, train_labels, train_noisy_labels, noise_type=None, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.train_data = list(train_data)
        self.train_labels = list(train_labels)
        self.train_noisy_labels = list(train_noisy_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.noise_type != 'clean':
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
