import os
import random
import warnings
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_cleaning import data_cleaning
from data_loader import feature_extraction_dataloader

test_size = 0.15
seed = 9
num_trees = 100
scoring = "accuracy"


def main():
    warnings.filterwarnings('ignore')
    # data_cleaning()
    loader = feature_extraction_dataloader()
    # loader.write_data()

    models = {'LR': LogisticRegression(max_iter=1000, random_state=seed, tol=0.001, C=1, warm_start=True),
              'LDA': LinearDiscriminantAnalysis(),
              'KNN': KNeighborsClassifier(),
              'CART': DecisionTreeClassifier(),
              'RF': RandomForestClassifier(n_estimators=num_trees, random_state=seed, warm_start=True),
              'NB': GaussianNB(),
              'SVM': SVC(random_state=seed)}

    data = sio.loadmat(os.path.join('data_set', 'img_feature.mat'))
    normalizer = MinMaxScaler(feature_range=(0, 1))
    img_feature = normalizer.fit_transform(data['image_feature'])
    name_label_map = data['names']
    print("[STATUS] feature vector normalized.")
    fv_trn, fv_tst, fv_l_trn, fv_l_tst = train_test_split(img_feature,
                                                          data['labels'][0],
                                                          test_size=test_size,
                                                          random_state=seed)
    print("[STATUS] split train and test data.")
    print("Train data   : {}".format(fv_trn.shape))
    print("Test data    : {}".format(fv_tst.shape))
    print("Train labels : {}".format(fv_l_trn.shape))
    print("Test labels  : {}".format(fv_l_tst.shape))

    results = []
    names = []

    for name in models.keys():
        model = models[name]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, fv_trn, fv_l_trn, cv=kf, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print("[CROSS_VAL_SCORE] Training {} : accuracy: {:.3f}, std: {:.3f}".format(name, cv_results.mean(), cv_results.std()))

    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison on training set')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Train and test:
    for key in models.keys():
        model = models[key]
        model.fit(fv_trn, fv_l_trn)
        pred = model.predict(fv_trn)
        result = np.equal(pred, fv_l_trn)
        accuracy = np.sum(result) / len(result)
        print("[RESULT] {} accuracy on training set: {:.03f}".format(key, accuracy))
        pred = model.predict(fv_tst)
        result = np.equal(pred, fv_l_tst)
        accuracy = np.sum(result) / len(result)
        print("[RESULT] {} accuracy on testing set: {:.03f}".format(key, accuracy))

    best_model = models['RF']

    # visulize:
    modeling_data_path = os.path.join('data_set', 'modeling_data')
    characters_folders = os.listdir(modeling_data_path)
    for character in characters_folders:
        pic_folder = os.path.join(modeling_data_path, character)
        all_pics = os.listdir(pic_folder)
        sampling = random.choices(all_pics, k=2)
        for pic in sampling:
            pic_dir = os.path.join(pic_folder, pic)
            image = cv2.imread(pic_dir)
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pic_feature = loader.extract_single_image(pic_dir)
            pic_feature = normalizer.transform(np.array([pic_feature]))
            # print(pic_feature.shape)
            pred_result = best_model.predict(pic_feature)[0]
            pred_result = name_label_map[pred_result]
            # print(pred_result)
            plt.imshow(display_image)
            plt.title("Prediction: {}, Truth: {}".format(pred_result, character))
            plt.show()


def test():
    modeling_data_path = os.path.join('data_set', 'modeling_data')
    characters_folders = os.listdir(modeling_data_path)
    for character in characters_folders:
        pic_folder = os.path.join(modeling_data_path, character)
        all_pics = os.listdir(pic_folder)
        sampling = random.choices(all_pics, k=2)
        for pic in sampling:
            pic_dir = os.path.join(pic_folder, pic)
            image = cv2.imread(pic_dir)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.show()


if __name__ == '__main__':
    main()
