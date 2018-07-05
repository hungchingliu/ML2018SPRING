#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:23:47 2018

@author: harry
"""
##### import #####
import csv
import numpy as np
import librosa
import librosa.display
import sys
##### import #####

##### const ######
pi = 3.141592653
width = 40
n_fil = 128
freq_dim = 1000
##### const ######


##### to run this function, you need to input 'audio_train_path'   #####
##### and make sure you have 'train.csv' in './data'               #####
##### this function will generate 'label_v.npy','train_data.npy'   #####
##### in same dic                                                  #####                  



def readfile():    
    v_name = []
    unv_name = []
    v_label = []
    unv_label = []    
    with open('./data/train.csv') as file:   
        line_n = 0
        reader = csv.reader(file)            
        for row in reader:   
            if line_n != 0:
                ## data ##
                if row[2] == 0:
                    ##unverified##
                    unv_name.append(row[0])
                    unv_label.append(row[1])
                else:
                    ##verified##                    
                    v_name.append(row[0])
                    v_label.append(row[1])
            line_n = line_n+1
    return v_name,v_label


def sampling(v_name,v_label,audio_train_path):
    print('sampling start')
    counter = 0
    train_data = np.zeros((len(v_name),width*2,n_fil))
    num = len(v_name)
    for i in range (0,num):
        print(v_label[i])
        y, sr = librosa.load(audio_train_path+'/'+ v_name[i])
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=16000)        
        sample = np.zeros((2*width,n_fil))
        spectrogram = (librosa.power_to_db(S,ref=np.max)).T
        max_index_r,max_index_c = np.where(spectrogram == np.max(spectrogram)) 
        max_index_r = max_index_r[0]
        if max_index_r >= width-1:
            sample[0:width] = spectrogram[max_index_r-width+1:max_index_r+1]
        else:
            temp = width-1 - max_index_r
            sample[temp:width] =  spectrogram[0:max_index_r+1]    
                
        total_row =  spectrogram.shape[0]
        if total_row - max_index_r >= width+1:
            sample[width:width*2] = spectrogram[max_index_r+1:max_index_r+width+1]
        else:
            temp = total_row - max_index_r
            sample[width:width+temp] = spectrogram[max_index_r :]	               
        train_data[counter] = sample 
        
        if counter%1 == 0:
            print('processing:',counter) 
        counter = counter + 1
       
    print('save file')    
    np.save('./temp/train_data',train_data)


if __name__ == '__main__':    
    audio_train_path = sys.argv[1]
    v_name,v_label = readfile()       
    sampling(v_name,v_label,audio_train_path)    
    np.save('./temp/label_v.npy',v_label)  
   
