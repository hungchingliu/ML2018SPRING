import h5py
import numpy as np
import csv
import keras
from keras.models import load_model
import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

def load_data():
    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        test_x = []
        iter = 0
        for line in reader:
            if iter == 0:
                iter += 1
                continue
            feature = line[1]
            feature = np.fromstring(feature, dtype = int, sep = ' ')
            feature = feature.reshape(48, 48)
            test_x.append(feature)

    test_x = np.array(test_x)
    test_x = test_x.astype('float32')
    test_x = test_x / 255.
    return test_x


def main():
    test_x = load_data()
    test_x = test_x.reshape(-1, 48, 48, 1)
    model = load_model('Ens_1.h5py')
    model.summary()
    test_y = model.predict(test_x)
    test_y = np.argmax(test_y, axis=1)
    print(test_y)

    with open(out_path, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for i in range(len(test_y)):
            writer.writerow([i, test_y[i]])

if __name__ == '__main__':
    main()
