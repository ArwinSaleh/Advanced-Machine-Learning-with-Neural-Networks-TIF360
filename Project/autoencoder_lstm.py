import pandas as pd
import numpy as np
import os
import pickle
from keras import metrics
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from helper_scripts.split_data import multivariate_data


window_length = 60
feats = 5

path_to_load = os.path.join(os.getcwd(), 'pkl_data' + os.sep + 'FAST.pkl')

print(path_to_load)
X = pickle.load(open(path_to_load, "rb"))

past_history = 60
future_target = 1
STEP = 1
np.random.seed(3)

train_split = 0.8
data_size = X.shape[0]

train_size = np.int(data_size * train_split)

x_train_single, y_train_single = multivariate_data(X, X[:, 3], 0,
                                                   train_size, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(X, X[:, 3],
                                               train_size, None, past_history,
                                               future_target, STEP,
                                               single_step=True)




class Autoencoder(Model):
   def __init__(self, window_length, feats):
      super(Autoencoder, self).__init__()
      self.encoder = tf.keras.Sequential([
         keras.layers.LSTM(128, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, feats), return_sequences=True, name='encoder_1'),
         keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'),
         keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3')])

      self.bridge = keras.layers.RepeatVector(window_length, name='bridge')

      self.decoder = tf.keras.Sequential([
         keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'),
         keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'),
         keras.layers.LSTM(128, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'),
         keras.layers.TimeDistributed(keras.layers.Dense(feats))])

   def call(self, x):
      encoded = self.encoder(x)
      bridged = self.bridge(encoded)
      decoded = self.decoder(bridged)
      return decoded


autoencoder = Autoencoder(window_length, feats)
autoencoder.compile(optimizer='adam', loss='mse')


callback = tf.keras.callbacks.EarlyStopping(
                                 monitor = "val_loss",
                                 min_delta = 0,
                                 patience = 5,
                                 verbose = 1,
                                 mode = "auto",
                                 baseline = None,
                                 restore_best_weights = True,
                                 )


# autoencoder = keras.models.load_model('auto')


autoencoder.fit(x_train_single, y_train_single,
                batch_size = 1,
                epochs = 1,
                shuffle = True,
                validation_data=(x_val_single, y_val_single),
                callbacks = [callback],
                )


Y = autoencoder(X)
print(Y.shape)
np.savetxt('Y.csv', tf.reshape(Y[:,:,-2], (90*60, 1)))
autoencoder.save('auto')
