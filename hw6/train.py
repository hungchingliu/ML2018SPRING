import pandas as pd
import numpy as np
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras import Input
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dot
from keras.layers import Add
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
"""
MF model is referred to the hands on tutorial of hw5 last semester
URL:https://docs.google.com/presentation/d/1oR3JJz7wVd5GD78AX9qImk3k-uS7cGZTGyxXeYraRlY/edit#slide=id.g1f620502c7_0_5

"""

batch_size = 256
epochs = 100
normalization = 1


def build_model(n_users, n_items, latent_dim=4):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])

    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Dropout(0.1)(user_vec)
    user_vec = Flatten()(user_vec)

    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Dropout(0.1)(item_vec)
    item_vec = Flatten()(item_vec)

    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)

    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)

    r_hat = Dot(axes=1)([user_vec, item_vec])

    # add bias
    r_hat = Add()([r_hat, user_bias, item_bias])

    model = keras.models.Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def main():
    dataset = pd.read_csv("dataset/train.csv")
    #dataset.UserID = dataset.UserID.astype('category').cat.codes.values
    #dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values
    dataset.UserID = dataset.UserID - 1
    dataset.MovieID = dataset.MovieID - 1

    #normalization
    if(normalization):
        mean = dataset.Rating.mean()
        std = dataset.Rating.std()
        np.save("mean.npy", mean)
        np.save("std.npy", std)
        dataset.Rating = (dataset.Rating - mean) / std



    print(dataset.head)
    n_users, n_movies = dataset.UserID.max() + 1, dataset.MovieID.max() + 1
    print(n_users, n_movies)
    model = build_model(n_users, n_movies)
    patience = 1
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
    train_model = model.fit([Dataset.UserID, Dataset.MovieID], Dataset.Rating, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping],
                            validation_split=0.1)
    model.save('MF.h5')



    """loss = train_model.history['loss']
    val_loss = train_model.history['val_loss']
    epoch = range(len(loss))

    plt.figure(1)
    plt.plot(epoch, loss, 'b', label='Training loss')
    plt.plot(epoch, val_loss, 'c', label='Validation loss')
    plt.legend()
    plt.savefig('NOR_fig.png')"""




if __name__ == "__main__":
    main()
