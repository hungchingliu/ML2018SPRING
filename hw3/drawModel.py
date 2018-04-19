import h5py
from keras.models import load_model
from keras.utils import plot_model



def main():

    model = load_model('model2.h5py')
    model.summary()
    plot_model(model, to_file='model.png', show_layer_names=False, show_shapes=True)

if __name__ == '__main__':
    main()