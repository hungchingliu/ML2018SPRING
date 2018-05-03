import numpy as np
from skimage import io
from skimage import transform
import os, sys
dir_path = sys.argv[1]
pic_path = sys.argv[2]

def load_data():
    dirs = os.listdir(dir_path)
    X=[]
    for file_name in dirs:
        img_path = os.path.join(dir_path, file_name)
        img = io.imread(img_path)
        #transform
        #img = transform.resize(img, (300,300))
        img = img.flatten()
        X.append(img)
    X = np.array(X)
    print(X.shape)
    return X

def load_picture():
    img_path = os.path.join(dir_path, pic_path)
    img = io.imread(img_path)
    img = img.flatten()
    return img

def main():
    X = load_data()
    X_mean = np.mean(X, axis=0)
    I = np.ones((415, 1))
    mean = np.dot(I, X_mean.reshape(1, 1080000))
    X = X - mean
    X = np.swapaxes(X, 0, 1)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U = np.swapaxes(U, 0, 1)
    pic1 = load_picture()
    pic1 = pic1.flatten()
    pic1 = pic1 - X_mean
    """w0 = np.dot(pic1, U[0])
    w1 = np.dot(pic1, U[1])
    w2 = np.dot(pic1, U[2])
    w3 = np.dot(pic1, U[3])
    reconstruct = w0 * U[0] + w1 * U[1] + w2 * U[2] + w3 * U[3]"""
    w = []
    for i in range(4):
        w.append(np.dot(pic1, U[i]))
    reconstruct = np.zeros((1080000, ))
    for i in range(4):
        reconstruct += w[i] * U[i]
    reconstruct = reconstruct + X_mean
    reconstruct = reconstruct.reshape(600, 600, 3)
    reconstruct -= np.min(reconstruct)
    reconstruct /= np.max(reconstruct)
    reconstruct = (reconstruct * 255).astype(np.uint8)
    io.imsave("reconstruction.png", reconstruct)

if __name__ == "__main__":
    main()
