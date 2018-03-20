import numpy as np
import csv
import math
import sys
###open csv
in_path = sys.argv[1]
csvfile = open(in_path, "r")
text = csv.reader(csvfile, delimiter = ',')
test = []
index = 0
for row in text:
    test.append([])
    for j in range(2,11):
        if row[j] == 'NR':
            test[index].append(float(0))
        else :
            test[index].append(float(row[j]))
    index += 1
test_x = []

for i in range(int(len(test) / 18)):
    test_x.append([])
    for k in range(9):
        for j in range(18):
            test_x[i].append(test[i * 18 + j][k])

###
for i in range(len(test_x)):
    for j in range(9):
        if test_x[i][j * 18 + 9] <= 0:
            if j > 0 and j < 8:
                test_x[i][j * 18 + 9] = (test_x[i][(j - 1) * 18 + 9] + test_x[i][(j + 1) * 18 + 9]) / 2
            elif j == 0:
                test_x[i][j * 18 + 9] = test_x[i][(j + 1) * 18 + 9]
            elif j == 8:
                test_x[i][j * 18 + 9] = test_x[i][(j - 1) * 18 + 9]
        if test_x[i][j * 18 + 8] <= 0:
            if j > 0 and j < 8:
                test_x[i][j * 18 + 8] = (test_x[i][(j - 1) * 18 + 8] + test_x[i][(j + 1) * 18 + 8]) / 2
            elif j == 0:
                test_x[i][j * 18 + 8] = test_x[i][(j + 1) * 18 + 8]
            elif j == 8:
                test_x[i][j * 18 + 8] = test_x[i][(j - 1) * 18 + 8]

test_x = np.array(test_x)
ones = np.ones((test_x.shape[0], 1))
test_x = np.concatenate((ones, test_x), axis = 1)

### load model
w = np.load('model_best.npy')

### write file
out_path = sys.argv[2]
ans = np.dot(test_x, w)
csv_out = open(out_path, "w+")
out_writer = csv.writer(csv_out, delimiter = ',', lineterminator = '\n')
out_writer.writerow(["id", "value"])
for i in range(len(ans)):
    out_writer.writerow(['id_' + str(i), ans[i][0]])
csv_out.close()
