import os
import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np

fixed_size = (224, 224)
path = os.path.join('data_set', 'modeling_data')
bins = 8

stat = "[STATUS]"
warn = "[WARNING]"


def hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def haralick_textures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)


def color_histo(image, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def main():
    character_labels = os.listdir(path)
    print(character_labels)

    features = []
    labels = []

    # loading images from folders
    for training_class in character_labels:
        image_folder = os.path.join(path, training_class)
        pics_name = os.listdir(image_folder)
        print("{} processing folder: {}".format(stat, training_class))
        for pic in pics_name:
            pic_dir = os.path.join(image_folder, pic)
            image = cv2.imread(pic_dir)
            image = cv2.resize(image, fixed_size)

            hu_value = hu_moments(image)
            haralick_value = haralick_textures(image)
            histogram_value = color_histo(image)

            image_features = np.hstack([hu_value, haralick_value, histogram_value])
            labels.append(training_class)
            features.append(image_features)
        print("{} finishing processing folder: {}".format(stat, training_class))

    print("{} completed Feature Extraction.".format(stat))
    print("{} feature vector shape: {}".format(stat, np.array(features).shape))
    print("{} feature vector shape: {}".format(stat, np.array(labels).shape))


if __name__ == '__main__':
    # image = cv2.imread('data_set/modeling_data/tohsaka_rin/doujin_002.png')
    # image = cv2.resize(image, fixed_size)
    # b, g, r = cv2.split(image)
    # image = cv2.merge((r, g, b))
    # plt.imshow(image)
    # plt.show()
    main()
