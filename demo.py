import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

veriler = pd.read_csv("EV-USA-Monthly.csv", sep=";")
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

veriler['Month_Year'] = veriler.iloc[:, 0].apply(lambda x: str(month_mapping[x.split('-')[0]]) + '.' + x.split('-')[1])

x = veriler[['Month_Year','GP','DP','EP','MRGDPI','TNL','TREP','PPI(BM)','CPI(NV)','NALR']]

A = veriler['PHEV']
A1= veriler['BEV']
A2= veriler['HEV']
y=(A+A1+A2)/3 #Output

y = y.replace({',': '.'}, regex=True)
x = x.replace({',': '.'}, regex=True)

X = x.iloc[41:,:]
Y = y.iloc[41:]

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_cnn_list= []
r2_cnn_list= []
mae_cnn_list= []
rmse_cnn_list= []
y_test_list=[]
y_cnn_head_list=[]

for train_index, test_index in kf.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Verileri normalize et
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(x_train_scaled.shape[1], 1)))
    cnn_model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(LSTM(units=50))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(500, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(200, activation='relu'))
    cnn_model.add(Dense(1))
    cnn_model.summary()
    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    cnn_model.fit(x_train_scaled.reshape((x_train_scaled.shape[0], x_train_scaled.shape[1], 1)), 
              y_train.values, 
              epochs=1000, 
              batch_size=16, 
              verbose=1)
  
    y_cnn_head = cnn_model.predict(x_test_scaled.reshape((x_test_scaled.shape[0], x_test_scaled.shape[1], 1)))

    mse_cnn = mean_squared_error(y_test.values, y_cnn_head)
    r2_cnn = r2_score(y_test.values, y_cnn_head)
    mae_cnn = mean_absolute_error(y_test.values, y_cnn_head)
    rmse_cnn = mean_squared_error(y_true=y_test.values,y_pred=y_cnn_head,squared=False)

    mse_cnn_list.append(mse_cnn)
    r2_cnn_list.append(r2_cnn)
    mae_cnn_list.append(mae_cnn)
    rmse_cnn_list.append(rmse_cnn)

    y_test_list.append(y_test.values)
    y_cnn_head_list.append(y_cnn_head)
    

avg_mse_cnn= np.mean(mse_cnn_list)
avg_r2_cnn= np.mean(r2_cnn_list)
avg_mae_cnn= np.mean(mae_cnn_list)
avg_rmse_cnn= np.mean(rmse_cnn_list)

print("CNN Model - Average Mean Squared Error:", avg_mse_cnn)
print("CNN Model - Average R-squared:", avg_r2_cnn)
print("CNN Model - MAE:", avg_mae_cnn)
print("CNN Model - RMSE:", avg_rmse_cnn)
