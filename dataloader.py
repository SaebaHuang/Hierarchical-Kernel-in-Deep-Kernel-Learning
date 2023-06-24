# encoding=UTF-8
from libsvm.svmutil import svm_read_problem
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Libs.uci_datasets import Dataset


"""
    Summarizing all dataloaders
"""


def get_dataloader(dataset_name):
    dataset_loader_dict = dict(
        # Moon scatter
        moons=moons_loader,
        # LIBSVM
        triazines=libsvm_loader,
        pyrim=libsvm_loader,
        bodyfat=libsvm_loader,
        mpg=libsvm_loader,
        # CIFAR-10
        cifar10=cifar10_loader,
        # UCI
        elevators=uci_loader,
        kin40k=uci_loader,
        servo=uci_loader,
    )
    dataset_loader = None

    if dataset_name in dataset_loader_dict.keys():
        dataset_loader = dataset_loader_dict[dataset_name]

    return dataset_loader

"""
    Scikit-learn demo datasets
"""
# Moons
def moons_loader(dataset_path, n_samples=1000):
    X, Y = make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=321)
    Y = 2 * Y - 1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=321, shuffle=True, stratify=Y)
    return X_train, X_test, Y_train, Y_test


"""
    LIBSVM regression tasks
"""


# libsvm loader
def libsvm_loader(dataset_path):
    (label, item) = svm_read_problem(dataset_path)

    X = np.asarray([list(item[i].values()) for i in range(len(item))])
    X = (2 * X - X.max(axis=0) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = np.nan_to_num(X)

    Y = np.asarray(label)
    Y = (2 * Y - Y.max() - Y.min()) / (Y.max() - Y.min())

    return X, Y

"""
    CIFAR-10
"""


# CIFAR-10
def cifar10_loader(dataset_path):
    import torch
    torch.manual_seed(321)

    # import cifar10 dataset with torchvision
    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=False)

    train_data = train_set.data / 255.0

    mean = train_data.mean(axis=(0, 1, 2))
    std = train_data.std(axis=(0, 1, 2))
    print('Normalize -- mean: {},  std: {}'.format(mean, std))

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=50000,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=10000,
        shuffle=False
    )

    X_train, Y_train = next(iter(train_loader))
    X_test, Y_test = next(iter(test_loader))

    X_train = X_train.numpy()
    Y_train = Y_train.numpy()

    X_test = X_test.numpy()
    Y_test = Y_test.numpy()

    X_train = np.reshape(X_train, newshape=(X_train.shape[0], -1))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], -1))

    return X_train, Y_train, X_test, Y_test

"""
    UCI loaders
"""


def uci_loader(dataset_name):
    data = Dataset(dataset_name)
    X = data.x
    Y = data.y

    # Normalization
    X = (2 * X - X.min(axis=0) - X.max(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = np.nan_to_num(X)

    # Reshape
    Y = np.reshape(Y, newshape=(-1,))
    Y = (2 * Y - Y.min() - Y.max()) / (Y.max() - Y.min())

    return X, Y

