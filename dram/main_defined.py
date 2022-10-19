import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

import utils
import svhn_loader as svhn_loader

from trainer import Trainer
from config import get_config


def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}
    # 32*32
    config.data_dir = "/home/ganche/Downloads/project/Visual-Attention-Model/SVHN/data/SVHN_multi.pickle"
    
    # for large dataset 64*64
    # config.data_dir = "/data2/share/ganche/SVHNClassifier/data"
    # "/home/ganche/Downloads/project/recurrent-visual-attention/mnist_digit_sample_8dsistortions9x9.npz"
    # instantiate data loaders
    if config.is_train:
        dloader = svhn_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            config.im_size,
            **kwargs,
        )
    else:
        dloader = svhn_loader.get_test_loader(
            config.data_dir, config.batch_size, config.im_size, **kwargs,
        )


    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
