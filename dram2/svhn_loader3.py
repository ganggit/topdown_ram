import numpy as np
from utils import plot_images
import pickle
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import Dataset

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        c, h, w = image.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h,
                      left: left + new_w]

        return image

def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    im_size = 32,
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


    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Grayscale(num_output_channels=1)
    ])
    
    path_to_train_lmdb_dir = data_dir + '/train.lmdb'
    train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir, transform),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)


    path_to_eval_lmdb_dir = data_dir + '/val.lmdb'
    valid_loader = torch.utils.data.DataLoader(
        Dataset(path_to_eval_lmdb_dir, transform),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, im_size, num_workers=4, pin_memory=False):
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
    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Grayscale(num_output_channels=1)
    ])
    path_to_test_lmdb_dir = data_dir + '/test.lmdb'
    data_loader = torch.utils.data.DataLoader(
        Dataset(path_to_test_lmdb_dir, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
