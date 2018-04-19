import numpy as np
import itertools
import csv
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
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
    train_x = train_x / 255.
    train_y = to_categorical(train_y)
    return train_x, train_y


def main():
    classes = ['angry', 'hate', 'fear', 'joy', 'sad', 'surprise', 'neutral']
    test_x, true_y = load_data()
    test_x = test_x.reshape(-1, 48, 48, 1)
    model = load_model('modelEns.h5py')
    model.summary()
    test_y = model.predict(test_x)
    test_y = np.argmax(test_y, axis=1)
    true_y = np.argmax(true_y, axis=1)
    print(true_y.shape, test_y.shape)
    print(true_y)
    print(test_y)
    cnf_matrix = confusion_matrix(true_y, test_y)
    print(cnf_matrix)

    title = 'Normalized Confusion Matrix'
    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.png')
if __name__ == '__main__':
    main()