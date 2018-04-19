import h5py
import numpy as np
import csv
import math
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


batch_size = 256
epochs = 300
n_class = 7

def load_data():
    with open('dataset/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        train_x = []
        train_y = []
        iter = 0
        for line in reader:
            if iter == 0:
                iter += 1
                continue
            label = line[0]
            feature = line[1]
            feature = np.fromstring(feature, dtype = int, sep = ' ')
            feature = feature.reshape(48, 48)
            train_x.append(feature)
            train_y.append(int(label))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.astype('float32')
    train_x = train_x / 255.
    train_y = to_categorical(train_y)
    return train_x, train_y

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(math.floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_valid = X_valid.reshape(-1, 48, 48, 1)
    return X_train, Y_train, X_valid, Y_valid

def build_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same', input_shape=(48, 48, 1)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), activation='linear', padding='same'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='linear'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    return model

def main():

    x_all, y_all = load_data()
    print(x_all.shape)
    train_x, train_label, valid_x, valid_label = split_valid_set(x_all, y_all, 0.1)
    model = build_model()
    print(train_x.shape, train_label.shape, valid_x.shape, valid_label.shape)



    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                                 zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)

    train_model = model.fit_generator(datagen.flow(train_x, train_label), steps_per_epoch=5*len(train_x)//batch_size, epochs=epochs, verbose=1, validation_data=(valid_x, valid_label))

    model.save('model5.h5py')
    acc = train_model.history['acc']
    val_acc = train_model.history['val_acc']
    loss = train_model.history['loss']
    val_loss = train_model.history['val_loss']
    epoch = range(len(acc))
    plt.figure(1)
    plt.plot(epoch, acc, 'b', label='Training acc')
    plt.plot(epoch, val_acc, 'r', label='Validation acc')
    plt.legend()
    plt.savefig('fig1.png')

    plt.figure(2)
    plt.plot(epoch, loss, 'b', label='Training loss')
    plt.plot(epoch, val_loss, 'r', label='Validation loss')
    plt.legend()
    plt.savefig('fig2.png')
if __name__ == '__main__':
    main()
