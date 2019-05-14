from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout  

model = Sequential()  

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  

model.add(Dropout(0.2)) 

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2))  

model.add(Dense(units = 1))  
model.compile(optimizer = 'adam', loss = 'mean_squared_error') 

model.fit(features_set, labels, epochs = 100, batch_size = 32) 

apple_testing_complete = pd.read_csv(r'E:\Datasets\apple_testing.csv')  
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values  
