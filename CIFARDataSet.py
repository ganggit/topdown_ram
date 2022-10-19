from numpy.core.fromnumeric import size
import torch

class CIFARDataSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, transform=None):
        'Initialization'
        self.labels = labels.astype(int)
        self.data = data
        self.transform = transform
        self.sz = len(data)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        index = index % self.sz
        # Select sample
        X = self.data[index]

        # Load data and get label
        if self.transform:
            X = self.transform(X)
        y = self.labels[index]

        return X, y