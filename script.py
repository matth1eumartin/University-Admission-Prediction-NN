import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

df = pd.read_csv("admissions_data.csv")


#splitting into features and labels
features = df.iloc[:, 1:8]
labels = df.iloc[:, -1]

#splitting into training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#scaling the data
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)
features_train_scaled = pd.DataFrame(features_train_scaled, columns = features_train.columns)
features_test_scaled = pd.DataFrame(features_test_scaled, columns = features_test.columns)

#creating the neural network model
def design_model(feature_data):
    model = Sequential()
    input = layers.InputLayer(input_shape = (features_train.shape[1],))
    model.add(input)
    hidden_layer_1 = layers.Dense(16, activation='relu')
    model.add(hidden_layer_1)
    model.add(layers.Dropout(0.1))
    hidden_layer_2 = layers.Dense(8, activation='relu')
    model.add(hidden_layer_2)
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    #initizialing the optimizer
    opt = Adam(learning_rate=0.005)
    model.compile(loss = 'mse', metrics = 'mae', optimizer = opt)
    return model

model = design_model(features_train_scaled)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

history = model.fit(features_train_scaled, labels_train.to_numpy(), epochs = 100, batch_size=8, verbose = 1, validation_split = 0.25, callbacks = [es])

val_mse, val_mae = model.evaluate(features_test_scaled, labels_test.to_numpy(), verbose = 0)

print("MAE: ", val_mae)

y_pred = model.predict(features_test_scaled)
print(f"model accuaracy: {r2_score(labels_test, y_pred)}")

#evaluating the model outputs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
fig.savefig("my_plots.png")