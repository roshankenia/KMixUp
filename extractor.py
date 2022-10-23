import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import optim, nn
from torchvision import models, transforms
from torch.autograd import Variable
import torchvision
from data.datasets import input_dataset
import argparse

import os
import sys
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type=str,
                    help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type=str,
                    help=' cifar10 or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)

# class FeatureExtractor(nn.Module):
#     def __init__(self, model):
#         super(FeatureExtractor, self).__init__()
#         # Extract VGG-16 Feature Layers
#         self.features = list(model.features)
#         self.features = nn.Sequential(*self.features)
#         # Extract VGG-16 Average Pooling Layer
#         self.pooling = model.avgpool
#         # Convert the image into one-dimensional vector
#         self.flatten = nn.Flatten()
#         # Extract the first part of fully-connected layer from VGG16
#         self.fc = model.classifier[0]

#     def forward(self, x):
#         # It will take the input 'x' until it returns the feature vector called 'out'
#         out = self.features(x)
#         out = self.pooling(out)
#         out = self.flatten(out)
#         out = self.fc(out)
#         return out


# # Initialize the model
# model = models.vgg16(pretrained=True)
# new_model = FeatureExtractor(model)


# define our pretrained resnet
model = torchvision.models.resnet152(pretrained=True).cuda()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
# remove last fully connected layer from model
model = torch.nn.Sequential(*(list(model.children())[:-1]))


# Change the device to GPU
# new_model = new_model.cuda()
new_model = model

# Will contain the feature
features = []
noisy_labels = []


#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human)

noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels),
      train_dataset.train_labels[:10])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)

print('extracting features')
x = 1
for i, (images, labels, indexes) in enumerate(train_loader):
    ind = indexes.cpu().numpy().transpose()
    batch_size = len(ind)

    images = Variable(images).cuda()
    labels = Variable(labels).cuda()
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the features from the images
        feature = new_model(images)
        # Convert to NumPy Array, Reshape it, and save it to features variable
    for j in range(len(feature)):
        features.append(feature[j].cpu().detach().numpy().reshape(-1))
        noisy_labels.append(labels[j].cpu().detach().numpy())

    x += 1
    if x == 10:
        break

# Convert to NumPy Array
features = np.array(features)
noisy_labels = np.array(noisy_labels)
print(features.shape)

print('clustering')
# Initialize the model
model = KMeans(n_clusters=num_classes)

# Fit the data into the model
model.fit(features)

# Extract the labels
labels = model.labels_

print(labels)  # [4 3 3 ... 0 0 0]

# calculate average actual label for each given label
averages = []
for i in range(num_classes):
    label_indices = np.where(labels == i)
    print()
    print(labels[label_indices])
    print(noisy_labels[label_indices])
    noisy_label_average = np.mean(noisy_labels[label_indices])
    averages.append(noisy_label_average)
print(averages)
