import pickle
import argparse
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.models import Model, Input, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint


MAX_LEN = 200


def build():
    input_shape = Input([20, MAX_LEN, 1])

    out = Conv2D(32, [4, 10], padding='same')(input_shape)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    # out = MaxPool2D([2, 2])(out)
    # out = Dropout(0.4)(out)

    out = Conv2D(32, [4, 10], strides=[1, 2], padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = MaxPool2D([2, 2])(out)
    out = Dropout(0.4)(out)

    out = Conv2D(64, [4, 10], strides=[1, 2], padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    # out = MaxPool2D([2, 2])(out)
    # out = Dropout(0.4)(out)

    out = Conv2D(64, [4, 10], strides=[1, 2], padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = MaxPool2D([2, 2])(out)
    out = Dropout(0.4)(out)

    out = Flatten()(out)

    out = Dense(64)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(0.4)(out)
    #
    # out = Dense(32)(out)
    # out = BatchNormalization()(out)
    # out = Activation('relu')(out)
    # out = Dropout(0.7)(out)

    out = Dense(41)(out)
    out = Activation('softmax')(out)

    model = Model(inputs=input_shape, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def datagen(x, y, batch_size):
    num = len(y)
    while True:
        xx = np.zeros([batch_size, 20, MAX_LEN, 1])
        ind = np.random.choice(num, batch_size, replace=False)
        for i, j, in enumerate(ind):
            # max_ind = np.argmax(x[j].sum(0))
            # left = max(0, max_ind - MAX_LEN//2)
            # piece = x[j][:, left:left+MAX_LEN].reshape([20, -1, 1])
            # xx[i, :, :piece.shape[1]] = piece
            xx[i, :, :x[j].shape[1]] = x[j][:, :MAX_LEN].reshape([20, -1, 1])
        yield xx, y[ind]


def score(y, preds):
    sc = np.zeros(41, dtype=np.float64)
    sc[:3] = np.array([1, 1/2, 1/3])
    count = 0
    for i in range(len(y)):
        ans = y[i]
        pred = preds[i]
        pred = np.argsort(pred)[::-1]
        count += np.sum((pred == ans) * sc)
    return count


def run(args):
    with open(args.xy, 'rb') as f:
        pk = pickle.load(f)
    data, labels = pk['data'], pk['labels']
    labels = to_categorical(labels, 41)

    idx = np.random.choice(len(data), len(data), replace=False)
    data = data[idx]
    labels = labels[idx]

    x_train = data[:args.train_num]
    x_valid = data[args.train_num:]
    y_train = labels[:args.train_num]
    y_valid = labels[args.train_num:]

    if args.semi:
        with open(args.semi, 'rb') as f:
            pk = pickle.load(f)
        s_data = pk['data']
        s_labels = to_categorical(pk['labels'], 41)
        idx = np.random.choice(len(s_data), 1000)
        s_data = s_data[idx]
        s_labels = s_labels[idx]
        x_train = np.concatenate([x_train, s_data])
        y_train = np.concatenate([y_train, s_labels])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True)]

    model = build()
    # model.summary()
    model.fit_generator(
        datagen(x_train, y_train, args.batch_size),
        epochs=args.epochs,
        steps_per_epoch=100,
        validation_data=datagen(x_valid, y_valid, args.batch_size),
        validation_steps=10,
        callbacks=callbacks
    )
    # model.save(args.model_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('xy')
    parser.add_argument('model_path')
    parser.add_argument('-i', type=int, default=100, dest='epochs')
    parser.add_argument('-b', type=int, default=200, dest='batch_size')
    parser.add_argument('-n', type=int, default=8000, dest='train_num')
    parser.add_argument('-s', dest='semi')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse())
    # f = lambda: None
    # for i in range(31, 40):
    #     setattr(f, 'xy', 'pickle/xy.pickle')
    #     setattr(f, 'model_path', 'mfcc/mfcc%d.mdn' % i)
    #     setattr(f, 'batch_size', 200)
    #     setattr(f, 'train_num', 8500)
    #     setattr(f, 'epochs', 100)
    #     setattr(f, 'semi', None)
    #     run(f)
