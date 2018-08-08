# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:10:08 2018

@author: Nidhi
"""

#Temporary driver module to test va module and dataset_loader
import os
os.environ['KERAS_BACKEND'] = 'theano'

from visual_attention_module import image_height, image_width
from sklearn import model_selection
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input, Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.utils.layer_utils import print_summary; 
from keras import optimizers, metrics
from dataset_loader import Dataset

#set path to local directories
project_directory = os.path.abspath('..')
model_backup_path =  os.path.join(project_directory,"data","backup")

'''
Load the training dataset: 
- Reads annotations from csv file 
- Sends each traning video to Visual Attention Module
- Appends each stshi sequence outputted to training inputs.
- Appends class and context labels as HOT encoded vectors to training outputs
'''
#define phase - training/testing
phase = "training"
#define size of temporal window 
duration = 40
#define length of stshi sequence
maxlen = 35

print("SETTING PHASE TO : ", phase, "\n\n")
ds = Dataset(phase, duration, maxlen)
print("\nDataset loaded.")

'''
Inputs to Memory Module:
X - Array of stshi sequences 
Y - Array of (1-HOT context vector, HOT-encoded class vector)
'''

#Training the memory module
hidden_dim = 64

X_all_stshiseq = ds.X_all_stshiseq
Y_all = ds.Y_all

input_dimensions = np.shape(X_all_stshiseq)
output_dimensions = np.shape(Y_all)

#DEBUG step
print ("Input dimensions : ",input_dimensions)
print ("Output dimensions : ", output_dimensions)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X_all_stshiseq,Y_all,test_size=0.2)
x_train = np.expand_dims(x_train, axis = 1)
x_test = np.expand_dims(x_test, axis = 1)
y_train = np.expand_dims(y_train, axis = 1)
y_test = np.expand_dims(y_test, axis = 1)

print("Training Input dimensions : ", np.shape(x_train))
print("Validation Input dimensions : ", np.shape(x_test))

print("Training Output dimensions : ", np.shape(y_train))
print("Validation Output dimensions : ", np.shape(y_test))

#THE MODEL
#Input - 1 stshi sequence is processed at a time, with each template flattened into a 1D vector
sequence = Input(shape=(1, input_dimensions[1], input_dimensions[2]), dtype='float64', name='input')

#create two shared lstms -
lstm_cell1 = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, return_sequences = True)
lstm_cell2 = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,  return_sequences = False)
#Make the lstm cells bidirectional + time distributed
#The first unit produces one output per stshi template
bi_lstm_unit1 = TimeDistributed(Bidirectional(lstm_cell1, merge_mode='concat', weights=None))(sequence)
#The second lstm outputs one one output for the whole sequence
bi_lstm_unit2 = TimeDistributed(Bidirectional(lstm_cell2, merge_mode='concat', weights=None))(bi_lstm_unit1)


#Dropout layer prevents overfitting
after_dp = Dropout(0.6)(bi_lstm_unit2)
#sigmoid activation layer
output = Dense(output_dimensions[1], activation='sigmoid', name='activation')(after_dp)

#Create mem_model from above defined layers
mem_model = Model(inputs=sequence, outputs=output)
#Your model is now two stacked bidirectional LSTM cell + a dropout layer + dense  layer+ an activation layer.
optimizers.Adam(lr=0.001, beta_1=0.6, beta_2=0.099, epsilon=1e-08, decay=0.005, clipnorm = 1., clipvalue = 0.5)
mem_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])

print('\n\nMemory module loaded. - Phase : ', phase, '\n\n')

print_summary(mem_model)

print('Beginning Training...')

#specify batch_size
batch_size = 1
#specify number of training epochs
num_epoch = 25

print("\n\nFitting to model.")
my_model = mem_model.fit(np.array(x_train), np.array(y_train), batch_size =batch_size, epochs=num_epoch, verbose =1, validation_data=[np.array(x_test), np.array(y_test)])

print("Model Training complete.")

#save the model
mem_model.save(os.path.join(model_backup_path,"model1.h5"))

print("Model saved to backup folder.")

