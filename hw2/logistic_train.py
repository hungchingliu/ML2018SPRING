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
    #print(ret)
    return ret

def crs_etrpy(fx, y):
    print(fx[0], y)
    fx = sigmoid(fx[0])
    print(fx, y)
    return -(y * math.log(fx) + (1 - y) * math.log(1 - fx))

def test(w, x_mean, x_var):
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
    ones = np.ones((test_x.shape[0], 1))
    test_x = np.concatenate((ones, test_x), axis=1)
    test_y = np.dot(test_x, w)
    for i in range(len(test_y)):
        test_y[i] = np.around(sigmoid(test_y[i]))
    with open('out.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id','label'])
        for i in range(len(test_y)):
            writer.writerow([i + 1, int(test_y[i][0])])




def main():
    train_x = parse_x()
    train_y = parse_y()
    x_mean = np.mean(train_x, axis = 0)
    x_var = np.var(train_x, axis = 0)
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            train_x[i][j] = (train_x[i][j] - x_mean[j]) / x_var[j]
    ones = np.ones((train_x.shape[0], 1))
    train_x = np.concatenate((ones, train_x), axis=1)
    w = []
    s_gra = []
    zero = []
    zero.append(0.)
    for i in range(len(train_x[0])):
        w.append(zero)
    w = np.array(w)
    for i in range(len(train_x[0])):
        s_gra.append(zero)
    w_zero = w

    l_rate = 10
    times = 500000
    w = w_zero
    x_tr = train_x.transpose()

    for i in range(times):
        hypo = sigmoid(np.dot(train_x, w))
        loss = hypo - train_y

        gra = np.dot(x_tr, loss)
        s_gra += (gra ** 2)
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        cost = np.sum(loss ** 2) / len(train_x)
        if i % 1000 == 0:
            hypo = np.around(hypo)
            valid = (train_y == hypo)
            print('iteration: %d | acc: %f' % (i, valid.sum() / len(train_y)))
    np.save('weight', w)
    np.save('x_mean', x_mean)
    np.save('x_var', x_var)


if __name__ == "__main__":
    main()
