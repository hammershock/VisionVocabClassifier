"""
you can download the zipped 15-Scene dataset from this website:
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12855452/15SceneImageDataset.rar


"""
import os
import pickle
from collections import defaultdict

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tqdm import tqdm

memory = joblib.Memory('./.cache')


@memory.cache(verbose=0)
def load_data(data_dir, split=150):
    data = defaultdict(list)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                category = os.path.split(root)[-1]
                fullpath = os.path.join(root, file)
                data[category].append(fullpath)
    for lst in data.values():
        lst.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    train_labels, val_labels = [], []
    train_paths, val_paths = [], []
    for category, paths in data.items():
        for i, path in enumerate(paths):
            (train_labels, val_labels)[i > split].append(category)
            (train_paths, val_paths)[i > split].append(path)

    return train_labels, val_labels, train_paths, val_paths


@memory.cache(verbose=0)
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kps, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def feature_histogram(descriptors: list[np.ndarray], kmeans_model):
    indices = kmeans_model.predict(descriptors)
    histogram = np.histogram(indices, bins=np.arange(len(kmeans_model.cluster_centers_) + 1))[0]
    histogram = histogram / np.sum(histogram)
    return histogram


if __name__ == '__main__':
    sift = cv2.SIFT_create(nOctaveLayers=5)
    # step 0: train-val split the dataset
    train_labels, val_labels, train_paths, val_paths = load_data('./15-Scene Image Dataset', split=150)

    # step 1. extract sift features for all images, both train and val data.
    # for each image, the shape of the descriptors is (N, 128); and we get a list of them
    train_descriptors = [extract_features(image_path) for image_path in tqdm(train_paths, desc='computing sift descriptors')]
    val_descriptors = [extract_features(image_path) for image_path in tqdm(val_paths, desc='computing sift descriptors')]

    try:
        with open('./kmeans.pkl', 'rb') as f:
            kmeans = pickle.load(f)

        train_feats = [feature_histogram(desc, kmeans) for desc in tqdm(train_descriptors, desc='building train feats')]
        val_feats = [feature_histogram(desc, kmeans) for desc in tqdm(val_descriptors, desc='building val feats')]
    except Exception:  # except any exceptions while loading and computing
        # step 2. cluster all descriptors of train data
        # Merge the descriptors from all train images, and cluster them. Any clusters are regarded as a visual word.
        kmeans = MiniBatchKMeans(n_clusters=700, random_state=42, verbose=0, batch_size=200)
        kmeans.fit(np.vstack(train_descriptors))
        for batch in tqdm([train_descriptors[i:i + 200] for i in range(0, len(train_descriptors), 200)], desc="Training KMeans"):
            data = np.vstack(batch)
            kmeans.partial_fit(data)

        # step 3. 将both训练集和测试集中的 descriptors归到最近的cluster，看作一个visual word, 根据每个图像各个word的占比计算的统计直方图特征
        train_feats = [feature_histogram(desc, kmeans) for desc in tqdm(train_descriptors, desc='building train feats')]
        val_feats = [feature_histogram(desc, kmeans) for desc in tqdm(val_descriptors, desc='building val feats')]
        with open('./kmeans.pkl', 'wb') as f:  # save the kmeans model trained.
            pickle.dump(kmeans, f)

    X_train = np.array(train_feats)
    X_val = np.array(val_feats)
    Y_train = np.array(train_labels)
    Y_val = np.array(val_labels)

    # 4. train a svm model to classify the images with the histogram features
    print('training svm')
    svm = SVC(kernel='rbf', C=10.0)
    svm.fit(X_train, Y_train)
    predictions = svm.predict(X_val)

    # 5. evaluate the model
    accuracy = accuracy_score(Y_val, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    # Accuracy: 60.05%
    cm = confusion_matrix(Y_val, predictions)
