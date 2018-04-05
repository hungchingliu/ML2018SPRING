import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
import csv


def parse_x(path):
    train_x = []
    with open(path, 'r') as file:
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

def parse_y(path):
    train_y = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in reader:
            train_y.append([])
            train_y[index].append(float(row[0]))
            index += 1
    train_y = np.array(train_y)
    return train_y

def test(x_mean, x_var, path):
    test_x = []
    with open(path, 'r') as file:
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
    return test_x

def acc(train_y, y):
    y = (y > 0.5)
    cnt = (y == train_y)
    return float(cnt.sum()) / len(y)

def main():
    in_path = sys.argv[3]
    out_path = sys.argv[4]
    train_in = sys.argv[1]
    train_label = sys.argv[2]
    train_x = parse_x(train_in)
    train_y = parse_y(train_label)
    x_mean = np.mean(train_x, axis=0)
    x_var = np.var(train_x, axis=0)
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            train_x[i][j] = (train_x[i][j] - x_mean[j]) / x_var[j]
    test_x = test(x_mean, x_var, in_path)
    times = 100
    result = np.zeros((len(test_x),))
    last_y = np.zeros((len(train_y),1))
    y = train_y

    for i in range(times):
        regr = DecisionTreeRegressor(max_depth=3)
        regr.fit(train_x, y)
        y = regr.predict(train_x)
        y = np.array(y)
        y = y.reshape(32561, 1)
        y_plus = y
        last_y += y_plus
        landa = acc(train_y, last_y)
        print('iter : %d acc : %f' %(i, landa))
        p_y = regr.predict(test_x)
        result += p_y
        y = train_y - last_y


    test_y = (result > 0.5)

    with open(out_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['id','label'])
            for i in range(len(test_y)):
                writer.writerow([i + 1, int(test_y[i])])
if __name__ == "__main__":
    main()
