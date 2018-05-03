import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from keras.models import load_model
import csv
import sys

path_train = sys.argv[1]
path_test = sys.argv[2]
path_out = sys.argv[3]

def load_data():
    with open(path_test, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        test_x = []
        iter = 0
        for line in reader:
            if iter == 0:
                iter += 1
                continue
            test = []
            test.append(int(line[1]))
            test.append(int(line[2]))
            test_x.append(test)
            iter += 1
        test_x = np.array(test_x)
        print(test_x)
        return test_x

def build_model():
    input_img = Input(shape=(784, ))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    encoder = Model(input = input_img, output=encoded)

    adam = Adam(lr=5e-4)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.summary()
    return encoder, autoencoder

def main():
    x = np.load(path_train)
    x = x.astype('float32') / 255.
    #train_num = 130000
    #print(x.shape)
    #train_x = x[:train_num]
    #valid_x = x[train_num:]

    """encoder, autoencoder = build_model()
    autoencoder.fit(train_x, train_x, epochs=1000, batch_size=256, shuffle=True, validation_data=(valid_x, valid_x))
    autoencoder.save('autoencoder.h5')
    encoder.save('encoder.h5')"""
    encoder = load_model('encoder.h5')
    encoder_imgs = encoder.predict(x)
    encoder_imgs = encoder_imgs.reshape(encoder_imgs.shape[0], -1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(encoder_imgs)

    test_x = load_data()
    same = []
    test_y = np.zeros(shape=test_x.shape)
    for i in range(test_x.shape[0]):
        test_y[i][0] = i
        a = kmeans.labels_[test_x[i][0]]
        b = kmeans.labels_[test_x[i][1]]
        if a == b:
            test_y[i][1] = 1
        else:
            test_y[i][1] = 0

    with open(path_out, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['ID', 'Ans'])
        for i in range(test_y.shape[0]):
            writer.writerow([int(test_y[i][0]), int(test_y[i][1])])

if __name__ == "__main__":
    main()
