# How To Reproduce
* keras version >= 2.1.6
* install librosa eg. pip install librosa
* for only ensemble predict data 
* bash ensemble.sh [audio_test_path] [answer_path]
* else
* bash reproduce.sh [audio_test_path] [answer_path]
* **it will download a model(400m) so it takes times**

# How To Train Method_1
* open src/method_1 and run 
* python readfile.py [audio_train_path]

> install matplotlib eg. pip install matplotlib
> || open 'train.py' and delete line 14, 109, 110 
* python train.py
* model will save as reproduce.h5 at same dic 


