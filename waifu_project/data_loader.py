import os
import time

import PIL
import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision as tv

stat = "[STATUS]"
warn = "[WARNING]"


class feature_extraction_dataloader:

    def __init__(self, fixed_size=(224, 224), data_path=os.path.join('data_set', 'modeling_data'), bins=8):
        self.path = data_path
        self.pic_size = fixed_size
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
        character_labels = os.listdir(self.path)
        print("{} Characters including {}".format(stat, character_labels))

        images_features = []
        images_rgb = []
        labels = []

        # loading images from folders
        for index, training_class in zip(range(len(character_labels)), character_labels):
            image_folder = os.path.join(self.path, training_class)
            pics_name = os.listdir(image_folder)
            print("{} {}/{} processing folder: {}".format(stat, index + 1, len(character_labels), training_class))
            for pic in pics_name:
                pic_dir = os.path.join(image_folder, pic)
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
        sio.savemat(os.path.join('data_set', file_name), data)
        print("{} save to data_set/img_feature.mat".format(stat))


class pytorch_dataloader():

    def __init__(self, data_dir=os.path.join('data_set', 'modeling_data'), size=(224, 224), mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.norm_para = {
            'train': [mean, std],
            'test': [mean, std]
        }
        self.data_dir = data_dir
        self.pic_size = size

    def gen_loader(self, data_transforms=None, batch_size=16, num_worker=4, train_proportion=0.8):
        if not data_transforms:
            data_transforms = tv.transforms.Compose([
                tv.transforms.RandomResizedCrop(self.pic_size),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(self.norm_para['train'][0], self.norm_para['train'][1])
            ])

        data_dir = 'data_set/modeling_data/'
        anime_data = tv.datasets.ImageFolder(data_dir, data_transforms)
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


class resnet_traditional_model:

    def __init__(self, img_transform=None, model=tv.models.resnet18(pretrained=True), mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean

        if std is None:
            self.std = [0.229, 0.224, 0.225]
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

    def feature_extraction(self):
        modeling_data_path = os.path.join('data_set', 'modeling_data')
        characters_folders = os.listdir(modeling_data_path)

        img_data = []
        labels = []

        since = time.time()
        for i, character in enumerate(characters_folders):
            print('[STATUS] {}/{} Processing {}'.format(i + 1, len(characters_folders), character))
            pic_folder = os.path.join(modeling_data_path, character)
            all_pics = os.listdir(pic_folder)
            for pic in all_pics:
                pic_dir = os.path.join(pic_folder, pic)
                raw_img = PIL.Image.open(pic_dir).convert('RGB')
                output = self.single_image_to_vec(raw_img)
                img_data.append(output)
                labels.append(character)
        time_elapsed = time.time() - since

        print('{} Feature extraction complete in {:.0f}m {:.0f}s'.format(stat, time_elapsed // 60, time_elapsed % 60))
        print("{} feature vector shape: {}".format(stat, np.array(img_data).shape))
        print("{} label vector shape: {}".format(stat, np.array(labels).shape))

        return img_data, labels

    def write(self, file_name='resnet_img_feature.mat'):
        img_features, img_labels = self.feature_extraction()
        names = np.unique(img_labels)
        encoder = LabelEncoder()
        target = encoder.fit_transform(img_labels)
        print("{} training labels encoded.".format(stat))
        data = {
            'resnet_feature': img_features,
            'labels': target,
            'names': names
        }
        filename = os.path.join('data_set', file_name)
        sio.savemat(filename, data)
        print("{} save to {}".format(stat, os.path.join(filename)))



if __name__ == '__main__':
    # image = cv2.imread('data_set/modeling_data/tohsaka_rin/doujin_002.png')
    # image = cv2.resize(image, fixed_size)
    # b, g, r = cv2.split(image)
    # image = cv2.merge((r, g, b))
    # plt.imshow(image)
    # plt.show()

    # loader = feature_extraction_dataloader()
    # # features, labels = loader.load_image_rgb()
    # loader.write_data()

    rn_tr = resnet_traditional_model()
    rn_tr.write()
