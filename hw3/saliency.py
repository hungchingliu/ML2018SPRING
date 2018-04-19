
from keras.models import load_model
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import numpy as np
import csv
import matplotlib.pyplot as plt



def load_data():
    with open('dataset/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        train_x = []
        train_y = []
        iter = 0
        for line in reader:
            if iter == 0:
                iter += 1
                continue
            label = line[0]
            feature = line[1]
            feature = np.fromstring(feature, dtype = int, sep = ' ')
            feature = feature.reshape(48, 48)
            train_x.append(feature)
            train_y.append(int(label))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.astype('float32')
    #train_x = train_x / 255.
    return train_x, train_y

def _shuffle(X):
    randomize = np.arange(X)
    np.random.shuffle(randomize)
    return (randomize[0:9])

def main():
    model = load_model('model2.h5py')
    layer_idx = -1
    train_x , train_y= load_data()
    train_x = train_x.reshape(-1, 48, 48, 1)
    path = 'saliencyMap/'
    label = ['angry/', 'hate/', 'fear/', 'joy/', 'sad/', 'surprise/', 'neutral/']
    #swap softmax activation function to linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    for i in range(len(train_x)):
        i_path = path + label[train_y[i]]
        class_idx = train_y[i]
        x = train_x[i].reshape(48, 48)

        plt.figure(1)
        plt.imshow(x, cmap='gray')
        plt.tight_layout()
        plt.savefig(i_path+'fig' + str(i) + '-1.png', dpi=100)

        plt.figure(2)
        plt.imshow(x, cmap='gray')
        grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=train_x[i])
        plt.imshow(grads, cmap='jet', alpha=0.5)
        plt.tight_layout()
        plt.savefig(i_path + 'fig' + str(i) + '-2.png', dpi=100)

        plt.figure(3)
        plt.imshow(grads, cmap='jet')
        plt.tight_layout()
        plt.savefig(i_path + 'fig' + str(i) + '-3.png', dpi=100)

        plt.figure(4)
        grads = grads.mean(axis=2)

        x[np.where(grads <= 70)] = np.mean(x)
        plt.imshow(x, cmap='gray')
        plt.tight_layout()
        plt.savefig(i_path + 'fig' + str(i) + '-4.png', dpi=100)

if __name__ == "__main__":
    main()