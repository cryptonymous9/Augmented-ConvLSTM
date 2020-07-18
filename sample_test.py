import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model import AugementedConvLSTM
import configparser
import h5py


Aug_ConvLSTM = AugementedConvLSTM
model = Aug_ConvLSTM().model()

print("\nModel:")
print(model.summary())

config = configparser.ConfigParser()
config.read('sample_config.ini')

def get_list_from_config(strr):
    req_list = [int(i) for i in strr.split(',')]
    return req_list

DIR = config.get('Paths', 'sample_dir')
model_weights = config.get('Paths', 'sample_weights')

model.load_weights(model_weights)


def load_dataset(X_=DIR+'X_low.npy', Y_=DIR+'Y_low.npy'):
    X = np.load(X_)
    Y = np.load(Y_)
    return X,Y

X, Y = load_dataset()
X = X.transpose(1,2,3,0)
Y = Y.reshape(-1,129,135,1)
print("X Shape: ", X.shape)
print("Y Shape: ", Y.shape)


time_steps = 5
generator = keras.preprocessing.sequence.TimeseriesGenerator(X, Y.reshape(-1,129,135,1),length=time_steps, batch_size=15)

pred_prec = model.predict_generator(generator, use_multiprocessing=False)


fig = plt.figure(figsize=(10,10))
X_train = X.transpose(0,3,1,2)[4:,0]
Y_train = Y[4:].reshape(-1, 129, 135)
pred_prec = pred_prec.reshape(-1,129,135)
for i in range(5):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(X_train[i]);axs[0].set_title("GCM Input");
    axs[1].imshow(pred_prec[i]);axs[1].set_title("Predicted Precipitation");
    axs[2].imshow(Y_train[i]);axs[2].set_title("Observed Precipitation");
    plt.show()
    print('----------------------------------------')