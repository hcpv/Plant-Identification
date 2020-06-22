from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, LeakyReLU, ReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


def CNN1(size, n_classes, dropout=0.3):
    model = Sequential()

    model.add(Input((size, size, 3)))
    model.add(Conv2D(32, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    plot_model(model, to_file='CNN.png')
    return model


def CNN2(size, n_classes, dropout=0.3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(size, size, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, (3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model
