from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def data_loader(normalize_para, resize, batch_size=12, num_worker=4, train_proportion=0.8):
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_para['train'][0], normalize_para['train'][1])
    ])
    data_dir = 'data_set/modeling_data/'
    anime_data = datasets.ImageFolder(data_dir, data_transforms)
    if train_proportion < 0.4 or train_proportion >= 1:
        print("Training set size not good! Set to default: 0.8")
        train_proportion = 0.8
    train_size = int(train_proportion * len(anime_data))
    test_size = len(anime_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(anime_data, [train_size, test_size])
    split_dataset = {'train': train_dataset, 'test': test_dataset}
    data_loaders = {
        x: torch.utils.data.DataLoader(split_dataset[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_worker)
        for x in ['train', 'test']
    }
    dataset_sizes = {x: len(split_dataset[x]) for x in ['train', 'test']}
    character_names = anime_data.classes
    print("Character names: {}".format(character_names))

    return data_loaders, dataset_sizes, character_names


def loader_display(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def main():
    plt.ion()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_para = {
        'train': [mean, std],
        'test': [mean, std]
    }
    resize = 224
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders, sizes, names = data_loader(norm_para, resize)
    inputs, classes = next(iter(loaders['train']))
    out = torchvision.utils.make_grid(inputs)

    char_names = [names[x] for x in classes]
    title = ""
    for name in char_names:
        title = title + name + "\n"
    loader_display(out, mean, std, title=title)


if __name__ == '__main__':
    main()
