import numpy as np
import sys
import csv

def sigmoid(x):
    x = np.clip(x, -500, 500)
    ret = 1 / (1.0 + np.exp(-x))
    return ret


def test(x_mean, x_var, mu1, mu2, shared_sigma, N1, N2, in_path, out_path):
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
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1) / N2)
    fx = np.dot(w, test_x.T) + b
    y = sigmoid(fx)
    test_y = np.around(y)

    with open(out_path, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id','label'])
        for i in range(len(test_y)):
            writer.writerow([i + 1, int(test_y[i])])
def main():
    mu1 = np.load('mu1.npy')
    mu2 = np.load('mu2.npy')
    shared_sigma = np.load('shared_sigma.npy')
    N1 = np.load('N1.npy')
    N2 = np.load('N2.npy')
    x_mean = np.load('x_mean.npy')
    x_var = np.load('x_var.npy')
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    test(x_mean, x_var, mu1, mu2, shared_sigma, N1, N2, in_path, out_path)

if __name__ == "__main__":
    main()