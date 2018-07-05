#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:06:41 2018

@author: harry
"""

######## import ########
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import text
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
########################

######## const ########
width = 40
n_fil = 128
######## const ########

######## to run this file, you need to have   ########
######## 'train_data.npy', 'label_v.npy'      ########
######## which can be generate by readfile.py ########
######## model will save as 'reproduce.h5'    ########

def train():
    all_data = np.load('./temp/train_data.npy')    
    all_label = np.load('./temp/label_v.npy')
    
    ##### generate label #####
    print('label generate start')
    tokenizer = text.Tokenizer(filters='\n') 
    tokenizer.fit_on_texts(all_label)
    
    all_label = tokenizer.texts_to_sequences(all_label)
    all_label = np.array(all_label,dtype='int')
    all_label = all_label-1
    all_label = np_utils.to_categorical(all_label)        
    print('label generate end')
    ##### generate label #####
    
    ##### suffle #####
    all_data = all_data.reshape(len(all_data),width*2, n_fil,1)
    
    index=list(range(0, len(all_data)))
    np.random.seed(1024)
    np.random.shuffle(index)
    all_data = all_data[index]
    all_label = all_label[index]
    ##### suffle #####
       
      
    ##### model structure #####
    
    ##### layer1 #####    
    model = Sequential()
    model.add( Conv2D( filters=48,kernel_size=(3,3),input_shape=(width*2, n_fil,1),padding='same'))
    model.add(Activation('relu')) 
    model.add( Conv2D( filters=48,kernel_size=(3,3),padding='same' ))    
    model.add(Activation('relu')) 
    model.add( Conv2D( filters=48,kernel_size=(3,3),padding='same' ))   
    model.add(Activation('relu')) 
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    ##### layer1 #####
    
    ##### layer2 #####    
    model.add( Conv2D( filters=96,kernel_size=(3,3),padding='same' ))
    model.add(Activation('relu'))   
    model.add( Conv2D( filters=96,kernel_size=(3,3),padding='same' ))
    model.add(Activation('relu'))      
    model.add( Dropout( rate=0.3 ) )
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    ##### layer2 #####
    
    ##### layer3 #####
    model.add( Conv2D( filters=192,kernel_size=(3,3),padding='same' ))
    model.add(Activation('relu')) 
    model.add( Conv2D( filters=192,kernel_size=(3,3),padding='same' ))
    model.add(Activation('relu'))     
    model.add( Dropout( rate=0.3 ) )
    model.add( MaxPooling2D( pool_size=(2,2) ) )
    ##### layer3 #####
    
        
    model.add( Flatten()  )      
    
    model.add( Dense(1024) )  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add( Dense(512) ) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add( Dense(41,activation='softmax')  )   
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
    ##### model structure #####

    history = model.fit(all_data,all_label,batch_size=200,
                      epochs=15,verbose=1,validation_split=0.05,
                      shuffle=True,initial_epoch=0)
    
    model.save('reproduce.h5')
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    
train() 