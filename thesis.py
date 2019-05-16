# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:48:25 2019

@author: Robot Hands
"""

from pandas import read_csv
import numpy
from numpy import sort

from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import accuracy_score

import warnings


warnings.filterwarnings("ignore", category = DeprecationWarning)



#load data
file = 'allData_16-24_05_2017.csv'
columns = ['Station 1 max height',
         'Station 1 min height',
         'Station 1 overflow',
         'Station 1 emergency switch',
         'Station 1 speed',
         'Station 1 health condition',
         'Station 2 max height',
         'Station 2 min height',
         'Station 2 overflow',
         'Station 2 emergency switch',
         'Station 2 pressure',
         'Station 2 health condition',
         'Station 3 max height',
         'Station 3 min height',
         'Station 3 overflow',
         'Station 3 emergency switch',
         'Station 3 pressure',
         'Station 3 health condition',
         'Station 4 container presence',
         'Station 4 suction time',
         'Station 4 RFID presence',
         'Station 4 emergency switch',
         'Station 4 pressure',
         'Station 4 health condition',
         'Station 1 health',
         'Station 2 health',
         'Station 3 health',
         'Station 4 health',
         'System Health',
         'System Condition']

df = read_csv(file, skiprows = 1, names = columns)
#array = df.values


S1_X = df.values[:, 0: 5]
S1_Y = df.values[:, 5]
   
S2_X = df.values[:, 6: 11]
S2_Y = df.values[:, 11]
   
S3_X = df.values[:, 12: 17]
S3_Y = df.values[:, 17]
    
S4_X = df.values[:, 18: 23]
S4_Y = df.values[:, 23]
"""    
    if prepro == "Normalize":
        scaler1 = Normalizer().fit(Raw_S1_X)
        S1_X = scaler1.transform(Raw_S1_X)
        
        scaler2 = Normalizer().fit(Raw_S2_X)
        S2_X = scaler2.transform(Raw_S2_X)
        
        scaler3 = Normalizer().fit(Raw_S3_X)
        S3_X = scaler3.transform(Raw_S3_X)
        
        scaler4 = Normalizer().fit(Raw_S4_X)
        S4_X = scaler4.transform(Raw_S4_X)
"""        

  
S1_X_S2_X = numpy.append(S1_X, S2_X, axis = 1)
S3_X_S4_X = numpy.append(S3_X, S4_X, axis = 1)
    
Sys_X = numpy.append(S1_X_S2_X, S3_X_S4_X, axis = 1)
Sys_Y = df.values[:, 23].astype('int')



seed = 0
size = .30

S1_X_model, S1_X_test, S1_Y_model, S1_Y_test = train_test_split(S1_X, S1_Y, test_size = size, random_state = seed)
S2_X_model, S2_X_test, S2_Y_model, S2_Y_test = train_test_split(S2_X, S2_Y, test_size = size, random_state = seed)
S3_X_model, S3_X_test, S3_Y_model, S3_Y_test = train_test_split(S3_X, S3_Y, test_size = size, random_state = seed)
S4_X_model, S4_X_test, S4_Y_model, S4_Y_test = train_test_split(S4_X, S4_Y, test_size = size, random_state = seed)
Sys_X_model, Sys_X_test, Sys_Y_model, Sys_Y_test = train_test_split(Sys_X, Sys_Y, test_size = size, random_state = seed)

S1_X_train, S1_X_valid, S1_Y_train, S1_Y_valid = train_test_split(S1_X_model, S1_Y_model, test_size = size, random_state = seed)
S2_X_train, S2_X_valid, S2_Y_train, S2_Y_valid = train_test_split(S2_X_model, S2_Y_model, test_size = size, random_state = seed)
S3_X_train, S3_X_valid, S3_Y_train, S3_Y_valid = train_test_split(S3_X_model, S3_Y_model, test_size = size, random_state = seed)
S4_X_train, S4_X_valid, S4_Y_train, S4_Y_valid = train_test_split(S4_X_model, S4_Y_model, test_size = size, random_state = seed)
Sys_X_train, Sys_X_valid, Sys_Y_train, Sys_Y_valid = train_test_split(Sys_X_model, Sys_Y_model, test_size = size, random_state = seed)


#Use XGBoost to show feature importance
model1 = xgb().fit(S1_X_train, S1_Y_train)
model2 = xgb().fit(S2_X_train, S2_Y_train)
model3 = xgb().fit(S3_X_train, S3_Y_train)
model4 = xgb().fit(S4_X_train, S4_Y_train)

plot_importance(model1)
plot_importance(model2)
plot_importance(model3)
plot_importance(model4)
pyplot.show()

S1_pred = model1.predict(S1_X_valid)
S2_pred = model2.predict(S2_X_valid)
S3_pred = model3.predict(S3_X_valid)
S4_pred = model4.predict(S4_X_valid)

S1_predictions = [round(value) for value in S1_pred]
S2_predictions = [round(value) for value in S2_pred]
S3_predictions = [round(value) for value in S3_pred]
S4_predictions = [round(value) for value in S4_pred]

accuracyS1 = accuracy_score(S1_Y_valid, S1_pred)
accuracyS2 = accuracy_score(S2_Y_valid, S2_pred)
accuracyS3 = accuracy_score(S3_Y_valid, S3_pred)
accuracyS4 = accuracy_score(S4_Y_valid, S4_pred)

thresholdS1 = sort(model1.feature_importances_)
thresholdS2 = sort(model2.feature_importances_)
thresholdS3 = sort(model3.feature_importances_)
thresholdS4 = sort(model4.feature_importances_)
print("-----------------------------------------------------------------------------")

for thresh in thresholdS1:
    selection1 = SFM(model1, threshold = thresh, prefit = True)
    select_X1_train = selection1.transform(S1_X_train)
    
    selection_model1 = xgb(objective = "multi:softmax", num_class = 3, )
    selection_model1.fit(select_X1_train, S1_Y_train)
    
    select_X1_test = selection1.transform(S1_X_valid)
    S1_pred = selection_model1.predict(select_X1_test)
    S1_predictions = [round(value) for value in S1_pred]
    accuracyS1 = accuracy_score(S1_Y_valid, S1_predictions)
    print("Thresh = %.3f, n=%d, Station 1 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X1_train.shape[1], accuracyS1 * 100))
    
"""   def extract_pruned_features(feature_importances, min_score = 200):
        column_slice = model1.feature_importances(feature_importances['weights'] > min_score)
        return column_slice.index.values
    
    pruned_features = extract_pruned_features(model1.feature_importances_, min_score = 200)
    S1_X_train_reduced = S1_X_train(pruned_features)
    S1_X_test_reduced = S1_X_test(pruned_features)
    
    def fit_and_print_metrics(S1_X_train, S1_Y_train, S1_X_valid, S1_Y_valid, model1):
        model1.fit()
"""   
print("-----------------------------------------------------------------------------")    

for thresh in thresholdS2:
    selection2 = SFM(model2, threshold = thresh, prefit = True)
    select_X2_train = selection2.transform(S2_X_train)
    selection_model2 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model2.fit(select_X2_train, S2_Y_train)
    
    select_X2_test = selection2.transform(S2_X_valid)
    S2_pred = selection_model2.predict(select_X2_test)
    S2_predictions = [round(value) for value in S2_pred]
    accuracyS2 = accuracy_score(S2_Y_valid, S2_predictions)
    print("Thresh = %.3f, n=%d, Station 2 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X2_train.shape[1], accuracyS2 * 100))
print("-----------------------------------------------------------------------------")

for thresh in thresholdS3:
    selection3 = SFM(model3, threshold = thresh, prefit = True)
    select_X3_train = selection3.transform(S3_X_train)
    selection_model3 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model3.fit(select_X3_train, S3_Y_train)
    
    select_X3_test = selection3.transform(S3_X_valid)
    S3_pred = selection_model3.predict(select_X3_test)
    S3_predictions = [round(value) for value in S3_pred]
    accuracyS3 = accuracy_score(S3_Y_valid, S3_predictions)
    print("Thresh = %.3f, n=%d, Station 3 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X3_train.shape[1], accuracyS3 * 100))
print("-----------------------------------------------------------------------------")

for thresh in thresholdS4:
    selection4 = SFM(model4, threshold = thresh, prefit = True)
    select_X4_train = selection4.transform(S4_X_train)
    selection_model4 = xgb(objective = "multi:softmax", num_class = 3)
    selection_model4.fit(select_X4_train, S4_Y_train)
    
    select_X4_test = selection4.transform(S4_X_valid)
    S4_pred = selection_model4.predict(select_X4_test)
    S4_predictions = [round(value) for value in S4_pred]
    accuracyS4 = accuracy_score(S4_Y_valid, S4_predictions)
    print("Thresh = %.3f, n=%d, Station 4 Accuracy (n = number of features used): %.2f%%" % (thresh, select_X4_train.shape[1], accuracyS4 * 100))
print("-----------------------------------------------------------------------------")
