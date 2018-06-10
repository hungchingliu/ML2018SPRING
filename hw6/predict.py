import numpy as np
import pandas as pd
from keras.models import load_model
import csv
import sys

normalization = 0

path_in = sys.argv[1]
path_out = sys.argv[2]

def main():
    test = pd.read_csv(path_in)
    #print(test.head)
    """dataset = pd.read_csv("dataset/train.csv")
    user = sorted(dataset.UserID.unique())
    movie = sorted(dataset.MovieID.unique())
    dataset.UserID = dataset.UserID.astype('category').cat.codes.values
    dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values
    userID = sorted(dataset.UserID.unique())
    movieID = sorted(dataset.MovieID.unique())
    userToID = {}
    movieToID = {}
    print(movie)
    print(movieID)
    print(len(user), len(userID))
    for i in range(len(user)):
        userToID[user[i]] = userID[i]
    for i in range(len(movie)):
        movieToID[movie[i]] = movieID[i]

    print(userToID)
    print(movieToID)
    test.UserID = test.UserID.apply(lambda x: userToID[x])
    test.MovieID = test.MovieID.apply(lambda x: movieToID[x])"""
    test.UserID = test.UserID - 1
    test.MovieID = test.MovieID - 1

    model = load_model("MF5.h5")
    model.summary()
    test_y = model.predict([test.UserID, test.MovieID])

    #normalization
    """if(normalization):
        mean = np.load("mean.npy")
        std = np.load('std.npy')
        test_y = test_y * std + mean"""

    test_y = np.clip(test_y, 0, 5)

    with open(path_out, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['TestDataID', 'Rating'])
        for i in range(len(test_y)):
            writer.writerow([i + 1, test_y[i][0]])

if __name__ == "__main__":
    main()