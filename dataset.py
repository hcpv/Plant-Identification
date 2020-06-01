import yaml
import os
import shutil
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
with open(os.path.join('config.yaml')) as stream:
    config = yaml.safe_load(stream)


def split_data(config):
    data_path = config["dataset"]["data_path"]
    test_data_path = config["dataset"]["test_data_path"]
    breakpoints = config["dataset"]["breakpoints"]
    images = []
    img_labels = []
    for i in range(1, len(breakpoints), 2):
        filename = str(breakpoints[i]) + '.jpg'
        label = int(i/2)
        image = Image.open(os.path.join(data_path, filename))
        images.append(np.array(image, dtype=np.uint8))
        img_labels.append(label)
    images = np.array(images)
    img_labels = np.array(img_labels)
    filename = config["dataset"]["test_h5"]
    file = h5py.File(os.path.join(test_data_path, filename), "w")
    file.create_dataset("X", np.shape(images), h5py.h5t.STD_U8BE, data=images,
                        compression="gzip", compression_opts=9)
    file.create_dataset("y", np.shape(img_labels), h5py.h5t.STD_U8BE,
                        data=img_labels, compression="gzip", compression_opts=9)
    file.close()


def displayImage(config):
    filename = config["dataset"]["test_h5"]
    test_data_path = config["dataset"]["test_data_path"]
    labels = config["labels"]
    file = h5py.File(os.path.join(test_data_path, filename), "r")
    images = file["X"]
    img_labels = file["y"]
    for image, img_label in zip(images, img_labels):
        plt.imshow(image)
        plt.title(labels[img_label])
        plt.show()
    file.close()


# convert_to_hdf5('data.h5', config)
# displayImage(config)
split_data(config)
