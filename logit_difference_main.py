# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import sys
import time
import math
import numpy as np
from data.cifar_index import CIFAR10Index
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
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--nt', type=float, default=0.3,
                    help='Noise threshold to divide noisy/clean')
# Adjust learning rate and for SGD Optimizer
# store starting time
begin = time.time()


def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_difference(logits, indexes, noise_or_not):
    diff_counts = np.zeros(10)
    noise_counts = np.zeros(10)
    for i in range(len(logits)):
        logit = sorted(F.softmax(logits[i]))
        # calculate difference between highest and second highest prediction
        diff = logit[-1] - logit[-2]
        # place count in our array
        diff_ind = math.floor(diff * 10)
        if diff_ind == 10:
            diff_ind = 9
        diff_counts[diff_ind] += 1

        noise_counts[diff_ind] += noise_or_not[indexes[i]]
    return diff_counts, noise_counts

# Train the Model


def train(epoch, train_loader, model, optimizer, noise_or_not):
    train_total = 0
    train_correct = 0

    diff_total = np.zeros(10)
    noise_total = np.zeros(10)

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        # calculate differences
        diff_counts, noise_counts = count_difference(
            logits, indexes, noise_or_not)
        diff_total = np.add(diff_total, diff_counts)
        noise_total = np.add(noise_total, noise_counts)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total += 1
        train_correct += prec
        loss = F.cross_entropy(logits, labels, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    train_acc = float(train_correct)/float(train_total)
    return train_acc, diff_total, noise_total


def clean_train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total += 1
        train_correct += prec
        loss = F.cross_entropy(logits, labels, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Clean Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    train_acc = float(train_correct)/float(train_total)
    return train_acc


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def noisy_train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # mixup data
        inputs, targets_a, targets_b, lam = mixup_data(images, labels)
        inputs, targets_a, targets_b = map(
            Variable, (inputs, targets_a, targets_b))

        # Forward + Backward + Optimize
        logits = model(inputs)

        prec_a, _ = accuracy(logits, targets_a, topk=(1, 5))
        prec_b, _ = accuracy(logits, targets_b, topk=(1, 5))

        prec = lam * prec_a + (1-lam)*prec_b
        # prec = 0.0
        train_total += 1
        train_correct += prec

        # mixup loss
        loss = lam * F.cross_entropy(logits, targets_a, reduce=True) + (
            1 - lam) * F.cross_entropy(logits, targets_b, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Noisy Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, A Training Accuracy: %.4F, B Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, prec_a, prec_b, loss.data))

    train_acc = float(train_correct)/float(train_total)
    return train_acc
# test
# Evaluate the Model


def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    # print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


def computeDifferences(logits):
    differences = []
    for i in range(len(logits)):
        logit = sorted(F.softmax(logits[i]))
        # calculate difference between highest and second highest prediction
        diff = logit[-1] - logit[-2]
        differences.append(diff)
    return differences


def findDivide(model, train_loader, noise_or_not, noise_threshold):
    model.eval()
    noisy_ind = []
    noisy_ind_noise = 0
    clean_ind = []
    clean_ind_noise = 0
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        # compute difference
        differences = computeDifferences(logits)

        # find those under threshold
        for j in range(len(differences)):
            if differences[j] < noise_threshold:
                noisy_ind.append(ind[j])
                noisy_ind_noise += noise_or_not[ind[j]]
            else:
                clean_ind.append(ind[j])
                clean_ind_noise += noise_or_not[ind[j]]
        print(f'Number of noisy samples: {len(noisy_ind)}')
        print(f'Noisy noise percentage: {(noisy_ind_noise/len(noisy_ind))}')
        print(f'Number of clean samples: {len(clean_ind)}')
        print(f'Clean noise percentage: {(clean_ind_noise/len(clean_ind))}')
        return clean_ind, noisy_ind


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
# load model
print('building model...')
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
model.cuda()


epoch = 0
clean_train_acc = 0
noisy_train_acc = 0

# training
# training
file = open('./checkpoint/%s_%s_nt_%d' %
            (args.dataset, args.noise_type, args.nt)+'_log_diff.txt', "w")
max_test = 0

noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # find clean/noisy
    clean_ind, noisy_ind = findDivide(
        model, train_loader, noise_or_not, args.nt)

    # create clean and noisy datasets
    clean_data = CIFAR10Index(train_dataset.train_data[clean_ind], train_dataset.train_labels[clean_ind], train_dataset.train_noisy_labels[clean_ind],
                              train_dataset.noise_type, train_dataset.transform, train_dataset.target_transform)
    noisy_data = CIFAR10Index(train_dataset.train_data[noisy_ind], train_dataset.train_labels[noisy_ind], train_dataset.train_noisy_labels[noisy_ind],
                              train_dataset.noise_type, train_dataset.transform, train_dataset.target_transform)
    # create data loaders
    clean_train_loader = torch.utils.data.DataLoader(dataset=clean_data,
                                                     batch_size=128,
                                                     num_workers=args.num_workers,
                                                     shuffle=True)
    noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_data,
                                                     batch_size=128,
                                                     num_workers=args.num_workers,
                                                     shuffle=True)

    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    # train clean then train noisy
    clean_train_acc = clean_train(epoch, clean_train_loader, model, optimizer)
    noisy_train_acc = noisy_train(epoch, noisy_train_loader, model, optimizer)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)

    if test_acc > max_test:
        max_test = test_acc

    # save results
    print('clean train acc on train images is ', clean_train_acc)
    print('noisy train acc on train images is ', noisy_train_acc)
    print('test acc on test images is ', test_acc)
    file.write("\nepoch: "+str(epoch))
    file.write("\tclean train acc on train images is " +
               str(clean_train_acc)+"\n")
    file.write("\tnoisy train acc on train images is " +
               str(noisy_train_acc)+"\n")
    file.write("\ttest acc on test images is "+str(test_acc)+"\n")
    file.flush()
file.write("\n\nfinal test acc on test images is "+str(test_acc)+"\n")
file.write("max test acc on test images is "+str(max_test)+"\n")

# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
file.write('Total runtime of the program is: ' + timeTaken)
file.flush()
file.close()
