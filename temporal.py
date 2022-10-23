# -*- coding:utf-8 -*-
from cProfile import label
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import sys
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


class TemporalLabels():

    def __init__(self, num_samples, num_classes, alpha=0.95, pre_train_length=5):
        # intialize our data arrays
        self.labels = torch.zeros(num_samples, num_classes)
        self.alpha = alpha
        self.pre_train_length = pre_train_length

    def addLabels(self, labels, indices, epoch):
        # for each sample update its label in our tensor
        rel_epoch = epoch - self.pre_train_length
        for i in range(len(indices)):
            index = indices[i]
            if rel_epoch > 0:
                self.labels[index] = self.alpha * self.labels[index] + \
                    (1-self.alpha)*labels[i]

                # bias correction
                cor = 1 - self.alpha**rel_epoch
                self.labels[index] = self.labels[index]/cor
            else:
                # base case
                self.labels[index] = labels[i]

        return self.labels[indices]
