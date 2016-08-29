# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 07:57:46 2016

@author: wx
"""
import time
from functools import wraps
from keras.models import Sequential
from keras.layers import LSTM
import csv
import numpy
import random
import xlrd
from keras.layers.core import Dropout, Activation,Dense
import matplotlib.pyplot as plt
from numpy import mat
from numpy import transpose
import numpy as np
from sklearn import metrics
import xlwt
from keras.optimizers import SGD
from keras.layers import GRU
import scipy.io as sio
import os  
def build_model():
    """
    build the model by using sequential
    return a model with 3 layers
    """
    numpy.random.seed(8)
    model=Sequential()
    #model.add(Dense(300,input_shape=(None,10,3),init='uniform',activation='relu'))
    #model.add(Dense(3, input_shape=(3,)))
    #model.add(Embedding(output_dim=(10,3,), input_dim=3, input_length=10))
    
   # model.add(Dropout(0.8))
    model.add(GRU(300,init='lecun_uniform', input_shape=(10,3), return_sequences=True,activation ='sigmoid'))
    #model.add(Embedding(output_dim=100,input_dim=300)) 
   # model.add(LSTM(30,init='uniform', return_sequences=True,activation ='sigmoid'))
    #model.add(Dense(8,init='uniform',activation='relu'))
    #model.add(Dense(4,init='uniform',activation='relu'))
    #model.add(Dense(100,init='uniform',activation='sigmoid'))
    
    #model.add(Dense(500,init='uniform',activation='sigmoid'))
    #model.add(GRU(10,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    #model.add(GRU(10,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    #.add(GRU(100,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    #model.add(GRU(100,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    model.add(GRU(100,init='lecun_uniform',return_sequences=False,activation ='sigmoid'))
    model.add(Dense(1,init='lecun_uniform'))
    sgd=SGD(lr=0.5,decay=0.01,momentum=0.2,nesterov=True)
    
    model.compile(loss='mean_absolute_error', optimizer='nadam', metrics=['accuracy'])
    return model    
def mkdir(path):
    
    import os

    path=path.strip()
 
    path=path.rstrip("\\")

    isExists=os.path.exists(path)


    if not isExists:
      
        print (path+" created")
        
        os.makedirs(path)
        return True
    else:
        
        print (path+' existed')
        return False
if __name__ == '__main__':
    i=0
    min1=4.0008
    max1=88.9919
    min2=-14.1366
    max2=16.4287
    min3=0
    max3=28.1513
    min4=0
    max4=28.1513                         
    data=sio.loadmat('trainset200000sample.mat') 
    trainset=data['trainset']
    data1=sio.loadmat('aimset200000sample.mat')
    aimset=data1['aimset']
    model = build_model() # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=300,batch_size=100,verbose=2,validation_split=0.2) 
    history=hist.history.items()
    mkdir('result0827')
    model.save_weights('result0827/300vprediction200000samplev1.h5')
    data2=sio.loadmat('valid10.mat')
    validset=data2['valid']    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 