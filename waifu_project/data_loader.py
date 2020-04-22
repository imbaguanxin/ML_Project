import os
from os import path

import PIL
import cv2
import mahotas
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from matplotlib import pyplot as plt
from scipy import io as sio
from sklearn.preprocessing import LabelEncoder

stat = "[STATUS]"
warn = "[WARNING]"


class WaifuDataLoader(object):

    def __init__(self, data_path=path.join('data_set', 'modeling_data'), pic_size=(224, 224)):
        self.data_path = data_path
        self.pic_size = pic_size

    def load_image(self):
        raise NotImplementedError("No implementation for DataLoaderBuilder.build()")


class FeatureExtractionDataLoader(WaifuDataLoader):

    def __init__(self, data_path=path.join('data_set', 'modeling_data'), pic_size=(224, 224), bins=8):
        super().__init__(data_path=data_path, pic_size=pic_size)
        self.bins = bins

    def hu_moments(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.HuMoments(cv2.moments(gray)).flatten()

    def haralick_textures(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return mahotas.features.haralick(gray).mean(axis=0)

    def color_histo(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_single_image(self, pic_dir, hu_moments=True, haralick=True, histogram=True):
        if not (hu_moments or haralick or histogram):
            raise ValueError("{} Must choose at lease one image model!".format(warn))
        else:
            image = cv2.imread(pic_dir)
            image = cv2.resize(image, self.pic_size)
            image_features = np.array([])
            if hu_moments:
                image_features = np.hstack((image_features, self.hu_moments(image)))
            if haralick:
                image_features = np.hstack((image_features, self.haralick_textures(image)))
            if histogram:
                image_features = np.hstack((image_features, self.color_histo(image)))
            return image_features

    def single_rgb(self, pic_dir):
        image = cv2.imread(pic_dir)
        image = cv2.resize(image, self.pic_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.array(image).flatten()

    def load_image(self, hu_moments=True, haralick=True, histogram=True, get_rgb=True):
        character_labels = os.listdir(self.data_path)
        print("{} Characters including {}".format(stat, character_labels))

        images_features = []
        images_rgb = []
        labels = []

        # loading images from folders
        for index, training_class in zip(range(len(character_labels)), character_labels):
            image_folder = path.join(self.data_path, training_class)
            pics_name = os.listdir(image_folder)
            print("{} {}/{} processing folder: {}".format(stat, index + 1, len(character_labels), training_class))
            for pic in pics_name:
                pic_dir = path.join(image_folder, pic)
                feature = self.extract_single_image(pic_dir,
                                                    hu_moments=hu_moments,
                                                    haralick=haralick,
                                                    histogram=histogram)
                labels.append(training_class)
                images_features.append(feature)
                if get_rgb:
                    rgb = self.single_rgb(pic_dir)
                    images_rgb.append(rgb)

        print("{} completed Feature Extraction.".format(stat))
        print("{} feature vector shape: {}".format(stat, np.array(images_features).shape))
        print("{} rgb vector shape: {}".format(stat, np.array(images_rgb).shape))
        print("{} label vector shape: {}".format(stat, np.array(labels).shape))
        return images_features, images_rgb, labels

    def write_data(self, file_name='img_feature.mat', hu_moments=True, haralick=True, histogram=True, rgb=False):
        img_features, img_rgb, img_labels = self.load_image(hu_moments=hu_moments,
                                                            haralick=haralick,
                                                            histogram=histogram,
                                                            get_rgb=rgb)
        names = np.unique(img_labels)
        encoder = LabelEncoder()
        target = encoder.fit_transform(img_labels)
        print("{} training labels encoded.".format(stat))
        data = {
            'image_feature': img_features,
            'image_rgb': img_rgb,
            'labels': target,
            'names': names
        }
        sio.savemat(path.join('data_set', file_name), data)
        print("{} save to data_set/img_feature.mat".format(stat))


def img_stat(data_dir, num_channels=3):
    cls_dirs = [d for d in os.listdir(data_dir) if path.isdir(path.join(data_dir, d))]
    channel_sum = np.zeros(num_channels)
    channel_sqr_sum = np.zeros(num_channels)
    pixel_num = 0

    for i, d in enumerate(cls_dirs):
        img_paths = [path.join(data_dir, d, img_file)
                     for img_file in os.listdir(path.join(data_dir, d))]
        print("{} Extract img color mean and std {}".format(stat, d))
        for img_path in img_paths:
            orig_img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img = rgb_img / 255.
            pixel_num += (img.size / num_channels)
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sqr_sum += np.sum(np.square(img), axis=(0, 1))

    img_mean = channel_sum / pixel_num
    img_std = np.sqrt(channel_sqr_sum / pixel_num - np.square(img_mean))

    return img_mean, img_std


class PyTorchDataLoader(WaifuDataLoader):

    def __init__(self, data_path=path.join('data_set', 'modeling_data'), pic_size=(224, 224),
                 channel_mean=None, channel_std=None):
        super().__init__(data_path=data_path, pic_size=pic_size)
        if channel_mean is None:
            # use mean from ImageNet
            channel_mean = [0.485, 0.456, 0.406]
        if channel_std is None:
            # use std from ImageNet
            channel_std = [0.229, 0.224, 0.225]
        self.norm_para = {
            'train': [channel_mean, channel_std],
            'test': [channel_mean, channel_std]
        }

    def load_image(self, data_transforms=None, batch_size=16, num_worker=4, train_proportion=0.8):
        if not data_transforms:
            data_transforms = tv.transforms.Compose([
                tv.transforms.RandomResizedCrop(self.pic_size),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(self.norm_para['train'][0], self.norm_para['train'][1])
            ])

        anime_data = tv.datasets.ImageFolder(self.data_path, data_transforms)
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

        return data_loaders, dataset_sizes, character_names, data_transforms

    def loader_display(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(self.norm_para['train'][0])
        std = np.array(self.norm_para['train'][1])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)


class ResNetTraditionalModel(WaifuDataLoader):

    def __init__(self, data_path=path.join('data_set', 'modeling_data'), pic_size=(224, 224),
                 img_transform=None, model=tv.models.resnet18(pretrained=True), mean=None, std=None):
        super().__init__(data_path, pic_size)
        mean_dataset, std_dataset = img_stat(data_path)
        if mean is None:
            self.mean = mean_dataset
        else:
            self.mean = mean

        if std is None:
            self.std = std_dataset
        else:
            self.std = std

        if img_transform is None:
            self.img_transform = tv.transforms.Compose([
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(self.mean, self.std)
            ])
        else:
            self.img_transform = img_transform
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        print("[STATUS] Using device: {}".format(self.device))

    def single_image_to_vec(self, raw_img):
        img = self.img_transform(raw_img)
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        outputs = self.model(img).detach().cpu().clone().numpy()[0]
        return outputs

    def load_image(self):
        characters_folders = os.listdir(self.data_path)

        img_data = []
        labels = []

        since = pd.to_datetime('now')
        for i, character in enumerate(characters_folders):
            print('[STATUS] {}/{} Processing {}'.format(i + 1, len(characters_folders), character))
            pic_folder = path.join(self.data_path, character)
            all_pics = os.listdir(pic_folder)
            for pic in all_pics:
                pic_dir = path.join(pic_folder, pic)
                raw_img = PIL.Image.open(pic_dir).convert('RGB')
                output = self.single_image_to_vec(raw_img)
                img_data.append(output)
                labels.append(character)
        time_elapsed = pd.to_datetime('now') - since

        print('{} Feature extraction complete in {:.0f}m {:.0f}s'
              .format(stat, time_elapsed.total_seconds() // 60, time_elapsed.total_seconds() % 60))
        print("{} feature vector shape: {}".format(stat, np.array(img_data).shape))
        print("{} label vector shape: {}".format(stat, np.array(labels).shape))

        return img_data, labels

    def write_data(self, file_name='resnet_img_feature.mat'):
        img_features, img_labels = self.load_image()
        names = np.unique(img_labels)
        encoder = LabelEncoder()
        target = encoder.fit_transform(img_labels)
        print("{} training labels encoded.".format(stat))
        data = {
            'resnet_feature': img_features,
            'labels': target,
            'names': names
        }
        filename = path.join('data_set', file_name)
        sio.savemat(filename, data)
        print("{} save to {}".format(stat, path.join(filename)))


if __name__ == '__main__':
    rn_tr = ResNetTraditionalModel()
    rn_tr.write_data()
