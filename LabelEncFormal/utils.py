import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms


# data loading and parameters for multi-test m1 and m2
def data_split(name):
    if name == 'abrupto':
        file = pd.read_csv('./data/abrupto/mixed_1010_abrupto_1.csv').values
        trait = file[:, :-1].astype(np.float64)
        label = file[:, -1].reshape(-1, 1)
        batch1, batch2, lr, wd, neuron = 256, 512, 0.1, 0.01, 4
    elif name == 'iris':
        file = pd.read_csv('./data/iris/iris.data', header=None).values
        trait = file[:, :-1].astype(float)
        label = file[:, -1].reshape(-1, 1)
        batch1, batch2, lr, wd, neuron = 16, 32, 0.1, 0.01, 4
    elif name == 'seeds':
        file = pd.read_table('./data/seeds/seeds_dataset.txt', sep='\\s+', header=None).values
        trait = file[:, :-1].astype(float)
        label = file[:, -1].reshape(-1, 1)
        batch1, batch2, lr, wd, neuron = 16, 32, 0.1, 0.01, 4
    elif name == 'drebin':
        file = np.load('./data/drebin/vec.npy')
        trait = file[:, :-1]
        label = file[:, -1].reshape(-1, 1)
        batch1, batch2, lr, wd, neuron = 256, 256, 0.1, 0.01, 4
    elif name == 'wine':
        file = pd.read_csv('./data/wine/wine.data', header=None).values
        trait = file[:, 1:]
        label = file[:, 0].reshape(-1, 1)
        batch1, batch2, lr, wd, neuron = 16, 32, 0.1, 0.01, 4
    elif name == 'cifar10':
        trait, label = cifar_data('cifar10')
        batch1, batch2, lr, wd, neuron = 32, 32, 0.01, 0.001, 128  # 0.41
    elif name == 'cifar100':
        trait, label = cifar_data('cifar100')
        batch1, batch2, lr, wd, neuron = 32, 32, 0.01, 0.001, 128  # 0.78
    elif name == 'purchase10':
        f = pd.DataFrame(pd.read_csv('./data/purchases_full/purchases_full.csv', index_col=0))
        trait = f.iloc[:, 5:].values
        label = np.atleast_2d(f['10classes'].values).T
        batch1, batch2, lr, wd, neuron = 128, 128, 0.01, 0.001, 128  # 0.95
    else:
        assert print("No dataset")
    return trait, label, batch1, batch2, lr, wd, neuron


# load cifar data
def cifar_data(name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    contents = os.listdir('./data/cifar10')

    if len(contents) == 0:
        if name == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
        elif name == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    else:
        if name == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)
        elif name == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=False, transform=transform)
            testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    train_data = next(iter(trainloader))
    test_data = next(iter(testloader))

    x_train, y_train = train_data
    x_test, y_test = test_data

    x_train_vectorized = x_train.view(x_train.size(0), -1)
    x_test_vectorized = x_test.view(x_test.size(0), -1)

    # transfer into numpy
    x_train_np = x_train_vectorized.numpy()
    x_test_np = x_test_vectorized.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    X_train = np.vstack((x_train_np, x_test_np))
    Y_train = np.hstack((y_train_np, y_test_np))

    return X_train, np.atleast_2d(Y_train).T




