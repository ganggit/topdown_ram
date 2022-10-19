import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from MNISTDataSet import MNISTDataSet

def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    # normalize = transforms.Normalize((0.1307,), (0.3081,)) # MNIST
    #normalize = transforms.Normalize((0.03606,), (0.01781,)) # clusttered MNIST
    normalize = transforms.Normalize((0.12,), (1,)) # clusttered MNIST
    # normalize = transforms.Normalize((0.03,), (1,)) # MNIST
    # normalize = transforms.Normalize((0.037,), (0.3081,)) # MNIST
    # normalize = transforms.Normalize((0,), (1,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = np.load(data_dir)

    X_train = dataset['X_train']
    y_train = dataset['y_train']
    num_train = len(X_train)
    indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices

    train_sampler = SubsetRandomSampler(train_idx)

    training_set = MNISTDataSet(X_train, y_train, trans)

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_set = MNISTDataSet(dataset['X_valid'], dataset['y_valid'], trans)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    normalize = transforms.Normalize((0.12,), (1,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    # dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    dataset = np.load(data_dir)
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    test_set = MNISTDataSet(X_test, y_test, trans)

    data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
