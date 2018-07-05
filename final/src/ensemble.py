import pickle
import sys
import numpy as np


def print_ans(preds, out_path, sample, id_name):
    waves = []
    with open(sample, 'r') as f:
        f.readline()
        for line in f:
            waves.append(line.split(',')[0])
    with open(id_name, 'rb') as f:
        id_name_map = pickle.load(f)

    with open(out_path, 'w') as f:
        f.write('fname,label\n')
        for i, wave in enumerate(waves):
            pred = np.argsort(preds[i])[:-4:-1]
            f.write(wave + ',')
            for s in pred:
                ins = id_name_map[s]
                f.write(ins + ' ')
            f.write('\n')


def run():
    dir_name = 'predict/'
    a = None
    with open('ensemble_config', 'r') as f:
        for line in f:
            line = line.split(' ')
            if a is None:
                a = np.load(dir_name + line[0]) * float(line[1])
            else:
                a += np.load(dir_name + line[0]) * float(line[1])
    b = np.load(dir_name + 'm1_predict.npy')
    id_name = np.load('method2/pickle/mfcc_id_name.pickle')
    b_id_name = np.load('method2/pickle/harry2.pickle')
    idx = [b_id_name[name] for name in [id_name[i] for i in range(41)]]
    b = b[:, idx]
    a += b * 5
    a += np.load(dir_name + 'mfcc10.npy') * 20
    print_ans(a, sys.argv[1], 'method_1/data/sample_submission.csv', 'method2/pickle/mfcc_id_name.pickle')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 ensemble.py <output_path>')
        exit(0)
    run()
