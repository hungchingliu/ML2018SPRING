import numpy as np
import sys
import csv
import math



def sigmoid(x):
    x = np.clip(x, -500, 500)
    ret = 1 / (1.0 + np.exp(-x))
    return ret


def test(w, x_mean, x_var, in_path, out_path):
    test_x = []
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            if index != 0:
                test_x.append([])
                for i in range(123):
                    test_x[index - 1].append(float(row[i]))
            index += 1
    test_x = np.array(test_x)
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            test_x[i][j] = (test_x[i][j] - x_mean[j]) / x_var[j]
    ones = np.ones((test_x.shape[0], 1))
    test_x = np.concatenate((ones, test_x), axis=1)
    test_y = np.dot(test_x, w)
    for i in range(len(test_y)):
        test_y[i] = np.around(sigmoid(test_y[i]))
    with open(out_path, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id','label'])
        for i in range(len(test_y)):
            writer.writerow([i + 1, int(test_y[i][0])])

def main():
    w = np.load('weight.npy')
    x_mean = np.load('x_mean.npy')
    x_var = np.load('x_var.npy')
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    test(w, x_mean, x_var, in_path, out_path)

if __name__ == "__main__":
    main()