#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:49:08 2018

@author: harry
"""
import csv
import numpy as np
import sys

from keras.models import load_model
from keras.preprocessing import text
import librosa


##### const ######
pi = 3.141592653
width = 40
n_fil = 128
##### const ######

#### load model ####
model = load_model('./data/fix.h5')
#### load model ####


##### to run this function, you need to input 'audio_test_path' #####
##### and make sure you have 'sample_submission.csv',           #####
##### 'label_all.npy', 'fix.h5'  in './data' in same dic        #####
##### this function will generate 'm1_predict.npy' in ../       #####

def generate_answer(audio_test_path):
    
    ##### generate label #####
    print('label generate start')
    dic_back = []
    all_label = np.load('./data/label_all.npy')
    tokenizer = text.Tokenizer(filters='\n',lower=False) 
    tokenizer.fit_on_texts(all_label)        
    for word in tokenizer.word_index:
        dic_back.append(word)
       
    ##### generate label #####
    
    ##### start predict #####
    print('starting predicting')
    line_n = 0
    answer = []    
    predict_array = []
    
    with open('./data/sample_submission.csv') as readfile:        
        reader = csv.reader(readfile)        
        
        for row in reader:
            if line_n > 0 :
                file_name = row[0]
                y, sr = librosa.load(audio_test_path+ '/' + file_name)
                
                if len(y) == 0:
                    answer.append([])
                    answer[-1].extend([file_name])
                    predict_str = dic_back[0]+' '+dic_back[1]+' '+dic_back[2]
                    answer[-1].extend([predict_str])
                    predict_array.extend(np.zeros((41)))
                else:    
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
                    
                    sample = sample.reshape(1,80,128,1)
                    predict = model.predict(sample)
                    predict_array.extend(predict)
                    
                    predict = predict[0].argsort()[-3:][::-1]     
                    
                    answer.append([])
                    answer[-1].extend([file_name])
                    predict_str = dic_back[predict[0]]+' '+dic_back[predict[1]]+' '+dic_back[predict[2]]
                    answer[-1].extend([predict_str])
                if line_n%10 == 0:
                    print('predicting line:',line_n)
            line_n = line_n + 1
            
    predict_array = np.array(predict_array)    
    np.save('../m1_predict',predict_array)  
    ##### start predict #####    

if __name__ == '__main__':    
    audio_test_path = sys.argv[1]
    generate_answer(audio_test_path)