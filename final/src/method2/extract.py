import os
import wave
import pickle
import argparse
import numpy as np
from librosa.feature import mfcc
from librosa.core import load


def get_fnames(folder):
    files = os.listdir(folder)
    files.sort()
    return files


def read_dir(folder):
    files = get_fnames(folder)
    data = []
    wave2ind = {}
    for i, file in enumerate(files):
        wave2ind[file] = i
        f = wave.open(os.path.join(folder, file), 'rb')
        raw = f.readframes(f.getnframes())
        f.close()
        seq = np.fromstring(raw, dtype=np.short)
        data.append(seq)
    data = np.array(data)
    return data, wave2ind


def extract(data):
    for i, row in enumerate(data):
        if len(row):
            data[i] = mfcc(row.astype(np.float64), sr=44100)
        else:
            print('error:', i)
            data[i] = np.zeros([20, 1])
        if i % 100 == 0:
            print(i, data[i].shape)
    return data


def read_label(train_label, id_name_file, wave2ind):
    with open(id_name_file, 'rb') as f:
        id_name_map = pickle.load(f)
    labels = np.zeros(len(wave2ind))
    with open(train_label, 'r') as f:
        f.readline()
        for line in f:
            line = line.split(',')
            labels[wave2ind[line[0]]] = id_name_map[line[1]]
    return labels


def run(args):
    data, wave2ind = read_dir(args.train_dir)
    data = extract(data)
    labels = read_label(args.train_label, args.id_name, wave2ind)
    print(labels[:10])
    dic = {'data': data, 'labels': labels}
    with open(args.out_path, 'wb') as f:
        pickle.dump(dic, f)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir')
    parser.add_argument('train_label')
    parser.add_argument('out_path')
    parser.add_argument('id_name')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse())
