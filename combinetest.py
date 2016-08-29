# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:26:40 2016

@author: WX
"""

import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
import tensorflow as tf
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
def mkdir(path):


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
def build_model3(number):
    """
    build the model by using sequential
    return a model with 3 layers
    """

    model=Sequential()
    with tf.device('/gpu:0'):
        model.add(GRU(number,init='lecun_uniform', input_shape=(10,3), return_sequences=True,activation ='sigmoid'))
    with tf.device('/gpu:1'):
        model.add(GRU(number,init='lecun_uniform',return_sequences=False,activation ='sigmoid'))
        model.add(Dense(1,init='lecun_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='nadam', metrics=['accuracy'])
    return model  
def build_model4(number1,number2):
    """
    build the model by using sequential
    return a model with 3 layers
    """
    model=Sequential()
    with tf.device('/gpu:0'):
        model.add(GRU(number1,init='lecun_uniform', input_shape=(10,3), return_sequences=True,activation ='sigmoid'))
        model.add(GRU(number1,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    with tf.device('/gpu:1'):   
        model.add(GRU(number2,init='lecun_uniform',return_sequences=False,activation ='sigmoid'))
        model.add(Dense(1,init='lecun_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='nadam', metrics=['accuracy'])
    return model  
def build_model5(number1,number2,number3):
    """
    build the model by using sequential
    return a model with 3 layers
    """
    model=Sequential()
    with tf.device('/gpu:0'):
        model.add(GRU(number1,init='lecun_uniform', input_shape=(10,3), return_sequences=True,activation ='sigmoid'))
        model.add(GRU(number1,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
    with tf.device('/gpu:1'):    
        model.add(GRU(number2,init='lecun_uniform',return_sequences=True,activation ='sigmoid'))
        model.add(GRU(number3,init='lecun_uniform',return_sequences=False,activation ='sigmoid'))
        model.add(Dense(1,init='lecun_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='nadam', metrics=['accuracy'])
    return model  
def load_data(choice):
    if choice==1:
        data=sio.loadmat('trainset10000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset10000sample.mat')
        aimset=data1['aimset']
    elif choice==2:
        data=sio.loadmat('trainset50000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset50000sample.mat')
        aimset=data1['aimset']
    elif choice==3:
        data=sio.loadmat('trainset100000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset100000sample.mat')
        aimset=data1['aimset']
    elif choice==4:
        data=sio.loadmat('trainset150000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset150000sample.mat')
        aimset=data1['aimset']
    elif choice==5:
        data=sio.loadmat('trainset200000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset200000sample.mat')
        aimset=data1['aimset']
    elif choice==6:
        data=sio.loadmat('trainset250000sample.mat') 
        trainset=data['trainset']
        data1=sio.loadmat('aimset250000sample.mat')
        aimset=data1['aimset']
    return trainset,aimset
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
    choice=1
    data2=sio.loadmat('valid10.mat')
    validset=data2['valid']
    mkdir('result0827')
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction10000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'10000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    choice=choice+1
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction50000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'50000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    choice=choice+1
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction100000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'100000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    choice=choice+1
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction150000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'150000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    choice=choice+1
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction200000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'200000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    choice=choice+1
    trainset,aimset=load_data(choice)
    model = build_model3(30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/30vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample30vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)
    
    model = build_model3(50) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/50vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample50vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(100) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/100vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample100vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(200) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/200vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample200vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model3(300) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/300vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample300vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 

    model = build_model4(30,30) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3030vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample3030vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(30,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/3010vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample3010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model4(10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/1010vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample1010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/101010vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample101010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction) 
    
    model = build_model5(10,10,5) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/10105vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample10105vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)

    model = build_model5(30,10,10) # Build Model(3 layers, LSTM)
    hist= model.fit(trainset,aimset,nb_epoch=500,batch_size=100,verbose=2,validation_split=0.2) 
    model.save_weights('result0827/301010vprediction250000samplev1.h5')    
    prediction=model.predict(validset)
    result_file = os.path.join(os.getcwd(),'result0827')
    result_file = os.path.join(result_file,'250000sample301010vprediction10to1v1.txt')
    numpy.savetxt(result_file, prediction)