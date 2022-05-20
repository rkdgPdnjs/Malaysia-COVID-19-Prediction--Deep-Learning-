# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:20:37 2022

@author: Alfiqmal
"""

# =============================================================================
# TRAIN SCRIPT
# =============================================================================

#%% PACKAGES

import os
import numpy as np
import pandas as pd
from modules import ExploratoryDataAnalysis, Visualize, DataPreprocess, ModelCreation
from sklearn.metrics import mean_absolute_error

#%% PATHS

TRAIN_PATH =  os.path.join(os.getcwd(), "dataset", "cases_malaysia_train.csv")
TEST_PATH = os.path.join(os.getcwd(), "dataset", "cases_malaysia_test.csv")

MODEL_PATH = os.path.join(os.getcwd(), "saved_models", "model.h5")
MMS_PATH = os.path.join(os.getcwd(), "saved_models", "mms.pkl")
LOG_PATH = os.path.join(os.getcwd(), "log")

#%% LOAD DATA

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

#%% ANALYZING DATA

train_df.info()
test_df.info()

#%% DATA INSPECTING AND CLEANING

eda = ExploratoryDataAnalysis()

train_df["cases_new"] = pd.to_numeric(train_df["cases_new"], errors = "coerce")
test_df["cases_new"] = pd.to_numeric(test_df["cases_new"], errors = "coerce")

train_df = train_df["cases_new"]
test_df = test_df["cases_new"]

train_df = eda.int_nan(train_df)
test_df = eda.int_nan(test_df)


#%% GRAPH PLOTTING

viz = Visualize()

viz.plotting_train(train_df)

#%% DATA PREPROCESSING (MINMAX SCALER)

dp = DataPreprocess()

mms, train_df_scaled, test_df_scaled = dp.minmaxscaler(train_df,
                                                       test_df, 
                                                       MMS_PATH)

window_size = 30

# train

X_train = []
y_train = []

for i in range(window_size, len(train_df)):
    X_train.append(train_df_scaled[i-window_size:i,0])
    y_train.append(train_df_scaled[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)


# test

temp = np.concatenate((train_df_scaled, test_df_scaled))
length_window = window_size + len(test_df_scaled)
temp = temp[-length_window:]

X_test = []
y_test = []

for i in range(window_size, len(temp)):
    X_test.append(temp[i-window_size:i, 0])
    y_test.append(temp[i,0])
    
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

#%% MODEL CREATION

mc = ModelCreation()

model = mc.lstm(X_train)

#%% MODEL TRAINING

mc.lstm_train(model, X_train, y_train, LOG_PATH)

#%% PERFORMANCE EVALUATION

predicted = []

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test, axis = 0)))
    
predicted = np.array(predicted)

#%% MODEL ANALYSIS

y_true = mms.inverse_transform(np.expand_dims(y_test, axis = -1))
y_pred = mms.inverse_transform(predicted.reshape(len(predicted), 1))

viz.plot_actual_predict(y_true, y_pred)

#%% MAPE CALCULATION

y_pred = y_pred.reshape(len(predicted), 1)

mean_absolute_error(y_true, y_pred)

print("MAPE for this model =",
      ((mean_absolute_error(y_true, y_pred)/sum(abs(y_true))))*100, "%")

#%% MODEL SAVE

model.save(MODEL_PATH)
