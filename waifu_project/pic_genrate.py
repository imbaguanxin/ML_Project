import torch
import torch.nn as nn


def our_net(num_of_class):
    return nn.Sequential(
        # 1
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 2
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        # 3 - 6
        nn.Conv2d(128, 128, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        # 7 - 11
        nn.Conv2d(128, 256, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        # 12 - 16
        nn.Conv2d(256, 512, 3, padding=1),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.Conv2d(512, 512, 3, padding=1),
        # 17
        nn.ReLU()

    )


if __name__ == '__main__':
    net = our_net()
    print(net)