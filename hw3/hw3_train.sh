#!/bin/bash

python3 cnn_gen.py $1 model2.h5py
python3 cnn_gen.py $1 model4.h5py
python3 ensemble.py
