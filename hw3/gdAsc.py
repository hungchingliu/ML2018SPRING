
from keras.models import load_model
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.pyplot as plt
from vis.input_modifiers import Jitter


def _shuffle(X):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize]

def main():
    model = load_model('model2.h5py')
    model.summary()
    # swap softmax activation function to linear
    layer_idx = -1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)


    layer_names = ['leaky_re_lu_1', 'leaky_re_lu_2', 'leaky_re_lu_3', 'leaky_re_lu_4']
    for lid in range(len(layer_names)):
        layer_name = layer_names[lid]
        layer_idx = utils.find_layer_idx(model, layer_name)
        filters = np.arange(get_num_filters(model.layers[layer_idx]))
        filters = _shuffle(filters)
        vis_images = []
        for idx in range(16):
            indices = filters[idx]
            img = visualize_activation(model, layer_idx, filter_indices=indices, tv_weight=0.,
                               input_modifiers=[Jitter(0.5)])
            vis_images.append(img)
        #img = img.reshape((48, 48))
        #plt.imshow(img, cmap="Blues")
        #plt.show()

        stitched = utils.stitch_images(vis_images, cols=8)
        plt.figure()
        plt.axis('off')
        shape = stitched.shape
        stitched = stitched.reshape((shape[0], shape[1]))
        plt.imshow(stitched)
        plt.title(layer_name)
        plt.tight_layout()
        plt.savefig('Filter_{}.png'.format(lid))

if __name__ == "__main__":
    main()