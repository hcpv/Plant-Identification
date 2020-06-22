import os
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess_required(config):
    return len(os.listdir(config["preprocessed_path"])) == 0


def save_data(X, y, path, filename, group, chunk=10):
    print('Saving data for '+group)
    X = np.array(X)
    y = np.array(y)
    print(group + ' data shape:')
    print(X.shape)
    print(y.shape)
    num_examples = X.shape[0]
    X_data = X[:chunk]
    y_data = y[:chunk]
    with h5py.File(os.path.join(path, filename), "a") as file:
        grp = file.create_group(group)
        X_dataset = grp.create_dataset("X", shape=X_data.shape, data=X_data, maxshape=(
            None, X_data.shape[1], X_data.shape[2], X_data.shape[3]), compression="gzip", compression_opts=9, chunks=True)
        y_dataset = grp.create_dataset("y", shape=y_data.shape, data=y_data, maxshape=(
            None,), compression="gzip", compression_opts=9, chunks=True)
        if num_examples > chunk:
            for i in range(chunk, num_examples, chunk):
                X_data = X[i:i+chunk]
                y_data = y[i:i+chunk]
                X_dataset.resize(X_dataset.shape[0]+X_data.shape[0], axis=0)
                X_dataset[-X_data.shape[0]:] = X_data
                y_dataset.resize(y_dataset.shape[0]+y_data.shape[0], axis=0)
                y_dataset[-y_data.shape[0]:] = y_data


def resize_append_X_y(X, y, img_path, img_label, size):
    image = Image.open(img_path)
    image = image.resize((size, size))
    X.append(np.array(image, dtype=np.uint8))
    y.append(img_label)
    return X, y


def preprocess_data(config):
    print("Preprocessing data")
    breakpoints = config["breakpoints"]
    raw_path = config["raw_path"]
    preprocessed_path = config["preprocessed_path"]
    preprocessed_h5 = config["preprocessed_h5"]
    val_size = config["train_val_split"]["val_size"]
    random_state = config["train_val_split"]["random_state"]
    img_size = config["img_size"]
    X = []
    y = []
    X_test = []
    y_test = []

    for i in range(1, len(breakpoints), 2):
        img_label = int(i/2)
        for j in range(breakpoints[i-1], breakpoints[i]):
            filename = str(j) + '.jpg'
            img_path = os.path.join(raw_path, filename)
            resize_append_X_y(X, y, img_path, img_label, img_size)
        filename = str(breakpoints[i]) + '.jpg'
        img_path = os.path.join(raw_path, filename)
        resize_append_X_y(X_test, y_test, img_path, img_label, img_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state)
    save_data(X_train, y_train, preprocessed_path, preprocessed_h5, 'train')
    save_data(X_val, y_val, preprocessed_path, preprocessed_h5, 'val')
    save_data(X_test, y_test, preprocessed_path, preprocessed_h5, 'test')


def display_img(config, group):
    filename = config["preprocessed_h5"]
    path = config["preprocessed_path"]
    labels = config["labels"]
    file = h5py.File(os.path.join(path, filename), "r")
    images = file[group+"/X"]
    img_labels = file[group+"/y"]
    print(images.shape)
    for image, img_label in zip(images, img_labels):
        plt.imshow(image)
        plt.title(labels[img_label])
        plt.show()
    file.close()


def load_train_val_data(config):
    print("Loading train-val data")
    preprocessed_path = config["preprocessed_path"]
    preprocessed_h5 = config["preprocessed_h5"]
    with h5py.File(os.path.join(preprocessed_path, preprocessed_h5), "r") as file:
        X_train = np.array(file["train/X"])
        y_train = np.array(file["train/y"])
        X_val = np.array(file["val/X"])
        y_val = np.array(file["val/y"])
    return np.array(X_train), y_train, X_val, y_val


def load_test_data(config):
    print("Loading test data")
    preprocessed_path = config["preprocessed_path"]
    preprocessed_h5 = config["preprocessed_h5"]
    with h5py.File(os.path.join(preprocessed_path, preprocessed_h5), "r") as file:
        X_test = np.array(file["test/X"])
        y_test = np.array(file["test/y"])
    return X_test, y_test
