import numpy as np
from utils import plot_images
import os,sys,pickle
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from CIFARDataSet import CIFARDataSet

version = sys.version_info

class CIFAR10Data(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, path):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')
        dataset ={}
        dataset["train_dataset"] = train_images
        dataset["train_labels"] = train_labels
        dataset["test_dataset"] = eval_images
        dataset["test_labels"] = eval_labels
        self.dataset = dataset

    @staticmethod
    def _load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict[b'data'].dtype == np.uint8
          image_data = data_dict[b'data']
          image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
          return image_data, np.array(data_dict[b'labels'])

    def get_train_valid_loader(
        self,
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


        # cifar = CIFAR10Data("/home/ganche/Downloads/project/cifar10_challenge/cifar10_data")
        X_train = self.dataset['train_dataset']
        y_train = self.dataset['train_labels']
        num_train = len(X_train)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx = indices[:num_train-split]
        val_idx = indices[-split:]
        train_sampler = SubsetRandomSampler(train_idx)

        training_set = CIFARDataSet(X_train[train_idx], y_train[train_idx], trans)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_set = CIFARDataSet(X_train[val_idx], y_train[val_idx], trans)
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


    def get_test_loader(self, batch_size, num_workers=4, pin_memory=False):
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
        # dataset = np.load(data_dir)
        X_test = self.dataset['test_dataset']
        y_test = self.dataset['test_labels']
        test_set = CIFARDataSet(X_test, y_test, trans)

        data_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return data_loader
