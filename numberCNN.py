#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:11:50 2017

@author: ethan
"""

#import MNIST handwriting data set
from keras.datasets import mnist
import keras as K

number_class = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[0]
from keras import backend
if backend.image_data_format() == 'channels_last':
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    shape = (28, 28, 1)
else:
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    shape = (1, 28, 28)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#convert class vectors to binary class matrices
y_train = K.utils.to_categorical(y_train, number_class)
y_test = K.utils.to_categorical(y_test, number_class)


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator




#initialzing Convolutional Neural Network
model = Sequential()

#first Convolutional layer with ReLU (rectified linear unit)
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=shape))
model.add(Activation('relu'))

#MaxPooling layer to retain spatial invariance
model.add(MaxPooling2D(pool_size=(2,2)))

#Second Convolutional Layer, ReLu Layer and Pooling Layer
model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Third Convolutional Layer, ReLu Layer and Pooling Layer
model.add(Conv2D(filters=128,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout to prevent overfitting
model.add(Dropout(0.25))

#Flattening the layers
model.add(Flatten())

#Full Connected Layer
model.add(Dense(128))
model.add(Activation('relu'))
#Dropout to prevent overfitting
model.add(Dropout(0.5))
#apply sigmoid to vector to assign probabilities
model.add(Dense(number_class))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


#Training the model
model.fit(x_train, y_train,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(x_test, y_test))

#evaluating CNN
result = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', result[0])
print('Test accuracy:', result[1])

from keras.models import model_from_json
#save the model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
print("saved the model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

