from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, LeakyReLU, ReLU, Input
from tensorflow.keras.models import Sequential


def ResNet(size, n_classes, dropout=0.3):
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(size, size, 3))
    resnet.trainable = False
    print(resnet.summary())

    model = Sequential()

    model.add(resnet)
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())

    return model
