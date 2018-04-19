from keras.models import load_model
from keras.models import Model
from keras.layers import Average, Input

def main():

    models = []
    temp = load_model('model2.h5py')
    temp.name = "model2"
    models.append(temp)


    temp = load_model('model4.h5py')
    temp.name = "model4"
    models.append(temp)

    model_input = Input(shape=models[0].input_shape[1:])
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)
    model_ensemble = Model(inputs=model_input, outputs=y)
    model_ensemble.summary()
    model_ensemble.save("Ens_1.h5py")

if __name__ == "__main__":
    main()
