import numpy as np
from utils import plot_images
import pickle
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
    normalize = transforms.Normalize((0.4,), (0.2,)) # clusttered MNIST
    # normalize = transforms.Normalize((0.03,), (1,)) # MNIST
    # normalize = transforms.Normalize((0.037,), (0.3081,)) # MNIST
    # normalize = transforms.Normalize((0,), (1,))
    #normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
    #                                 std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), normalize])

    # load dataset
    # dataset = np.load(data_dir)
    with open(data_dir+'/train.pkl', 'rb') as f:
    	dataset = pickle.load(f)
    X_train = dataset[0]
    y_train = dataset[1]
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
    with open(data_dir+'/val.pkl', 'rb') as f:
    	dataset = pickle.load(f)  
    valid_set = MNISTDataSet(dataset[0], dataset[1], trans)
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
    normalize = transforms.Normalize((0.4,), (0.2,))
    #normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
    #                                 std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    trans = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.ToTensor(), normalize])

    # load dataset
    # dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    with open(data_dir+'/test.pkl', 'rb') as f:
    	dataset = pickle.load(f) 
    X_test = dataset[0]
    y_test = dataset[1]
    test_set = MNISTDataSet(X_test, y_test, trans)

    data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader