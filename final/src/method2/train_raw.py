import argparse
import numpy as np
from keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D
from keras.layers import Dense, Dropout, GlobalMaxPool1D
from keras.models import Input, Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from extract import *
# from pretrained import get_1d_conv_model, Config
# import pretrained

MAX_LEN = 2 * 16000


def build():
    input_shape = Input([MAX_LEN, 1])

    out = Conv1D(16, 9)(input_shape)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv1D(16, 9)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPool1D(16)(out)
    out = Dropout(0.1)(out)

    out = Conv1D(32, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv1D(32, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPool1D(4)(out)
    out = Dropout(0.1)(out)

    out = Conv1D(32, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv1D(32, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPool1D(4)(out)
    out = Dropout(0.1)(out)

    out = Conv1D(256, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv1D(256, 3)(out)
    # out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = GlobalMaxPool1D()(out)
    out = Dropout(0.2)(out)

    out = Dense(64, activation='relu')(out)
    out = Dense(128, activation='relu')(out)
    out = Dense(41, activation='softmax')(out)

    model = Model(input_shape, out)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    return model


def datagen(x, y, batch_size):
    num = len(y)
    while True:
        ind = np.random.choice(num, batch_size, replace=False)
        # yield x[ind] * np.random.normal(1, 0.04, x[ind].shape), y[ind]
        yield x[ind], y[ind]


def run(args):
    model = build()

    folder = args.train_dir
    files = get_fnames(folder)
    wave2ind = {}
    x = np.zeros([len(files), 32000])
    for i, file in enumerate(files):
        wave2ind[file] = i
        # raw, _ = load(os.path.join(folder, file), sr=16000, res_type='kaiser_fast')
        # raw = pretrained.audio_norm(raw)
        # x[i][:len(raw)] = raw[:32000]
        # print(i, file)
    # x = np.expand_dims(x, axis=-1)
    # np.save('pickle/sr16000.pickle', x)
    x = np.load('pickle/sr16000.pickle.npy')

    # config = pretrained.Config(sampling_rate=16000, audio_duration=2, n_folds=10, learning_rate=0.001)
    # model = pretrained.get_1d_conv_model(config)

    # data, wave2ind = read_dir(args.train_dir)
    # x = np.empty([len(data), MAX_LEN, 1])
    # for i, d in enumerate(data):
    #     x[i, :len(d), 0] = d[:MAX_LEN]
    labels = read_label(args.train_label, args.id_name, wave2ind)
    labels = to_categorical(labels, 41)
    x_train = x[:args.train_num]
    x_valid = x[args.train_num:]
    y_train = labels[:args.train_num]
    y_valid = labels[args.train_num:]

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min'),
                 ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True)]

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
    parser.add_argument('train_dir')
    parser.add_argument('train_label')
    parser.add_argument('id_name')
    parser.add_argument('model_path')
    parser.add_argument('-i', type=int, default=100, dest='epochs')
    parser.add_argument('-b', type=int, default=100, dest='batch_size')
    parser.add_argument('-n', type=int, default=8000, dest='train_num')
    parser.add_argument('-s', dest='semi')
    return parser.parse_args()


if __name__ == '__main__':
    # run(parse())
    f = lambda: None
    for k in range(7, 10):
        setattr(f, 'train_dir', 'data/audio_train')
        setattr(f, 'train_label', 'data/train.csv')
        setattr(f, 'id_name', 'pickle/mfcc_id_name.pickle')
        # setattr(f, 'model_path', 'model/1d_conv%d.mdn' % k)
        setattr(f, 'model_path', 'tmp')
        setattr(f, 'batch_size', 64)
        setattr(f, 'train_num', 8500)
        setattr(f, 'epochs', 100)
        setattr(f, 'semi', None)
        run(f)
