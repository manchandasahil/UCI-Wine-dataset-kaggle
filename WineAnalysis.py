# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:27:46 2017

@author: Sahil Manchanda
"""
import pandas as pd
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

wine_data = pd.read_csv('winequality-data.csv')
#print(wine_data.describe())
y = wine_data['quality'].values


from keras.utils import np_utils
print(y)
y = np_utils.to_categorical(y)
print(y)
del(wine_data['quality'])
del(wine_data['id'])
wine_data = wine_data.values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(wine_data,y,test_size = 0.25,random_state=42)

model = Sequential()
model.add(Dense(40,input_dim = 11,kernel_initializer = 'glorot_normal',activation = 'relu'))
model.add(Dense(30,kernel_initializer="normal",activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(15,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(10,kernel_initializer="normal",activation = 'softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(wine_data,y,epochs=1600, verbose=1)
#y_pred = model.predict(x_test)
#print(model.evaluate(x_test,y_test))

test = pd.read_csv("winequality-solution-input.csv")
del(test['id'])

#import xlsxwriter
#
#workbook = xlsxwriter.Workbook('sample_submission.xlsx')
#worksheet = workbook.add_worksheet()
#Y_t = Y_t[Y_t>0]
b = list()
y_pred = model.predict(x_test)
(m,n) = y_pred.shape
for i in range(m):
    b.append(np.argmax(y_pred[i]))

#col = 1
#row = 1
#for data in enumerate(b):
#    worksheet.write_column(row, col, data)
b = np.array(b).astype(int)
np.savetxt('out.csv',b, delimiter=',')
