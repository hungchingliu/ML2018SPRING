import numpy as np
import sys
import csv
import math

def parse_x():
    train_x = []
    with open('train_x.csv', 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            if index != 0:
                train_x.append([])
                for i in range(123):
                    train_x[index - 1].append(float(row[i]))
            index += 1
    train_x = np.array(train_x)
    return train_x

def parse_y():
    train_y = []
    with open('train_y.csv', 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            train_y.append([])
            train_y[index].append(float(row[0]))
            index += 1
    train_y = np.array(train_y)
    return train_y

def sigmoid(x):
    x = np.clip(x, -500, 500)
    ret = 1 / (1.0 + np.exp(-x))
    return ret

def test(x_mean, x_var, mu1, mu2, shared_sigma, N1, N2):
    test_x = []
    with open('dataset/test_X.csv', 'r') as file:
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
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1) / N2)
    fx = np.dot(w, test_x.T) + b
    y = sigmoid(fx)
    test_y = np.around(y)

    with open('out_gen.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id','label'])
        for i in range(len(test_y)):
            writer.writerow([i + 1, int(test_y[i])])

def main():
    train_x = parse_x()
    train_y = parse_y()
    x_mean = np.mean(train_x, axis = 0)
    x_var = np.var(train_x, axis = 0)
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            train_x[i][j] = (train_x[i][j] - x_mean[j]) / x_var[j]
    size = len(train_x)
    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((len(train_x[0]), ))
    mu2 = np.zeros((len(train_x[0]), ))
    for i in range(size):
        if train_y[i] == 1:
            mu1 += train_x[i]
            cnt1 += 1
        else:
            mu2 += train_x[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((len(train_x[0]), len(train_x[0])))
    sigma2 = np.zeros((len(train_x[0]), len(train_x[0])))
    for i in range(size):
        if train_y[i] == 1:
            sigma1 += np.dot(np.transpose([train_x[i] - mu1]), [train_x[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([train_x[i] - mu2]), [train_x[i] - mu2])
    sigma1 /= cnt1
    sigma2 /= cnt2


    shared_sigma = (float(cnt1) / size) * sigma1 + (float(cnt2) / size) * sigma2
    np.save('mu1', mu1)
    np.save('mu2', mu2)
    np.save('shared_sigma', shared_sigma)
    np.save('N1', cnt1)
    np.save('N2', cnt2)


if __name__ == "__main__":
    main()
