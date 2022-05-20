# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:20:43 2022

@author: Alfiqmal
"""

# =============================================================================
# MODULES SCRIPT
# =============================================================================


#%% PACKAGES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import datetime
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import plot_model



#%% 

class ExploratoryDataAnalysis():
    
    def init(self):
        
        pass
    
    def int_nan(self, data):
        
        data = data.interpolate(method = "linear").astype("int")
        return data


class Visualize():
    
    def plotting_train(self, data):
       
        plt.figure()
        plt.plot(data)
        return plt.show()
    
    def plot_actual_predict(self, data1, data2):
        
        plt.figure()
        plt.plot(data1)
        plt.plot(data2)
        plt.legend(["Actual", "Predicted"])
        return plt.show()
    
class DataPreprocess():
    
    def minmaxscaler(self, data1, data2, path):
        
        mms = MinMaxScaler()
        data1 = mms.fit_transform(np.expand_dims(data1, axis = -1))
        data2 = mms.transform(np.expand_dims(data2, axis = -1))
        pickle.dump(mms, open(path, "wb"))
        return mms, data1, data2
    
class ModelCreation():
    
    def lstm(self, data):
        model = Sequential()
        model.add(LSTM(64,
                       activation = "tanh",
                       return_sequences = True,
                       input_shape = (data.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.summary()

        plot_model(model)

        model.compile(optimizer = "adam",
                      loss = "mse",
                      metrics = ["mse"])
        
        return model
    
    def lstm_train(self, model, data1, data2, path):
        
        log_files = os.path.join(path,
                         datetime.datetime.now().strftime("%Y%m%d - %H%M%S"))

        tensorboard_callback = TensorBoard(log_dir = log_files, histogram_freq = 1)

        return model.fit(data1, data2, 
                         epochs = 50,
                         batch_size = 128,
                         callbacks = [tensorboard_callback])
    

