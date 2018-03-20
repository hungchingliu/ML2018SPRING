import pandas as pd
import numpy as np
import csv
import math


df = pd.read_csv('train.csv', encoding='big5')

data = []
for month in range(12):
    data.append([])
    for day in range(20):
        for time in range(24):
            data[month].append([])
            for dirt in range(18):
                if df.iloc[month * 20 * 18 + day * 18 + dirt, time + 3] == "NR":
                    data[month][day * 24 + time].append(float(0))
                else:
                    data[month][day * 24 + time].append(float(df.iloc[month * 20 * 18 + day * 18 + dirt, time + 3]))

x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        y.append([])
        for k in range(9):
            x[i * 471 + j].append(data[i][j + k])
        y[i * 471 + j].append(data[i][j + 9][9])

train_x = []
train_y = []
index = 0
for h in range(12):
    for i in range(471):
        det = h * 471 + i
        if det >= 1200 and det < 1250:
            continue
        if det >= 1400 and det < 1450:
            continue
        train_x.append([])
        train_y.append([])
        train_y[index].append(y[h * 471 + i][0])
        for j in range(9):
            for k in range(18):
                train_x[index].append(x[h * 471 + i][j][k])
        index += 1
for i in range(len(train_x)):
    for j in range(9):
        if train_x[i][j * 18 + 9] <= 0:
            if j > 0 and j < 8:
                train_x[i][j * 18 + 9] = (train_x[i][(j - 1) * 18 + 9] + train_x[i][(j + 1) * 18 + 9]) / 2
            elif j == 0:
                train_x[i][j * 18 + 9] = train_x[i][(j + 1) * 18 + 9]
            elif j == 8:
                train_x[i][j * 18 + 9] = train_x[i][(j - 1) * 18 + 9]
        if train_x[i][j * 18 + 8] <= 0:
            if j > 0 and j < 8:
                train_x[i][j * 18 + 8] = (train_x[i][(j - 1) * 18 + 8] + train_x[i][(j + 1) * 18 + 8]) / 2
            elif j == 0:
                train_x[i][j * 18 + 8] = train_x[i][(j + 1) * 18 + 8]
            elif j == 8:
                train_x[i][j * 18 + 8] = train_x[i][(j - 1) * 18 + 8]
train_x = np.array(train_x)
train_y = np.array(train_y)


ones = np.ones((train_x.shape[0], 1))
x_max = np.max(train_x, axis = 0)
x_mean = np.mean(train_x, axis = 0)
x_var = np.var(train_x, axis = 0)
np.save('mean', x_mean)
np.save('var', x_var)
for i in range(len(train_x)):
    for j in range(len(train_x[0])):
        train_x[i][j] = (train_x[i][j] - x_mean[j]) / x_var[j]
train_x = np.concatenate((ones, train_x), axis=1)

w = []
zero = []
zero.append(0.)
for i in range(len(train_x[0])):
    w.append(zero)
w = np.array(w)
l_rate = 0.0025
times = 100000

x_tr = train_x.transpose()
for i in range(times):
    Xw = np.dot(train_x, w)
    err = train_y - Xw
    grad = -2 * np.dot(x_tr, err) / len(train_x)
    MSE = np.sum(err**2) / len(train_x)
    RMSE = math.sqrt(MSE)
    w = w - l_rate * grad
    if(i % 1000 == 0):
        print('iter: %d | loss : %f ' %(i, RMSE))

Xw = np.dot(train_x, w)
for i in range(len(Xw)):
    if Xw[i] < 0:
        Xw[i] = -Xw[i]
err =  Xw - train_y
MSE = np.sum(err**2) / len(train_x)
RMSE = math.sqrt(MSE)
print(RMSE)


#### save model
np.save('model', w)
