import yaml
import numpy as np
from dataset import *
from models import *
from visualize import *
from utils import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

with open(os.path.join('config.yaml')) as stream:
    config = yaml.safe_load(stream)

if(preprocess_required(config["dataset"])):
    preprocess_data(config["dataset"])

img_size = config["dataset"]["img_size"]
n_classes = len(config["dataset"]["labels"])
model_name = config["train"]["model_name"]
output_path = config["output_path"]
saved_models_path = config["saved_models_path"]
batch_size = config["train"]["batch_size"]
epochs = config["train"]["epochs"]
X_train, y_train, X_val, y_val = load_train_val_data(config["dataset"])

X_train = X_train.astype('float32')
X_train /= 255

X_val = X_val.astype('float32')
X_val /= 255

train_size = X_train.shape[0]
val_size = X_val.shape[0]

y_train = to_categorical(y_train, n_classes)
y_val = to_categorical(y_val, n_classes)

train_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                               shear_range=0.3, height_shift_range=0.1, zoom_range=0.1)
val_gen = ImageDataGenerator()

train_generator = train_gen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_gen.flow(X_val, y_val, batch_size=batch_size)

model = ResNet(img_size, n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training started")
# history = model.fit(X_train, y_train, epochs=config["train"]["epochs"],
#                     batch_size=config["train"]["batch_size"], validation_data=(X_val, y_val))

with tf.device('/CPU:0'):
    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=train_size //
                        batch_size, validation_data=val_generator, validation_steps=val_size//batch_size)
print("Training completed")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

path = output_path + '/' + model_name + '_accuracy.png'
line_plot(acc, val_acc, path, 'training accuracy', 'validation accuracy', model_name)

path = output_path + '/' + model_name + '_loss.png'
line_plot(loss, val_loss, path, 'training loss', 'validation loss', model_name)

y_val_pred = model.predict(X_val, batch_size=config["train"]["batch_size"])
y_val_pred = np.argmax(y_val_pred, axis=1)
y_val = np.argmax(y_val, axis=1)

accuracy, precision, recall, f1 = classification_report(y_val, y_val_pred)

with open(os.path.join(output_path, model_name + '.txt'), 'w') as file:
    file.write('Accuracy' + ":" + str(accuracy) + '\n')
    file.write('Precision' + ":" + str(precision) + '\n')
    file.write('Recall' + ":" + str(recall) + '\n')
    file.write('F1 score' + ":" + str(f1) + '\n')

path = saved_models_path + '\\' + model_name + '.h5'
model.save(path)
