import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

humidity_training_complete = pd.read_csv('data/humidity-2018.csv')
humidity_training_processed = humidity_training_complete.iloc[:, 1:2].values
temperature_training_complete = pd.read_csv('data/temperature-2018.csv')
temperature_training_processed = temperature_training_complete.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))

humidity_training_scaled = scaler.fit_transform(humidity_training_processed)
temperature_training_scaled = scaler.fit_transform(temperature_training_processed)

features_set = []  
labels = []  
for i in range(60, 1260):  
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])
    
features_set, labels = np.array(features_set), np.array(labels)  

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
