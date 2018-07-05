########## How To Reproduce
###### keras version >= 2.1.6
#### install librosa eg. pip install librosa
git clone https://github.com/hungchingliu/ML2018SPRING.git  clone final project
git clone https://gitlab.com/harry1003/ml_final.git         clone model
#### Add 'fix.h5' in 'ml_final' to final/src/method_1/data
bash reproduce.sh [audio_test_path]


########## How To Train Method_1
open src/method_1 and run 
python readfile.py [audio_train_path]

#### install matplotlib eg. pip install matplotlib
#### or open 'train.py' and delete line 14, 109, 110 
python train.py
model will save as reproduce.h5 at same dic 


