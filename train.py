import numpy as np
import pickle
import matplotlib.pyplot as plt

import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing as prep

from sklearn.model_selection import train_test_split

from model import AugementedConvLSTM
import configparser

import h5py

config = configparser.ConfigParser()
config.read('config.ini')

DIR = config.get('Paths', 'dir')

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def get_list_from_config(strr):
    req_list = [int(i) for i in strr.split(',')]
    return req_list

config = configparser.ConfigParser()
config.read('config.ini')

DIR = config.get('Paths', 'dir')

DIR_monsoon_gcm = config.get('Paths', 'processed_monsoon_gcm')
DIR_monsoon_observed = config.get('Paths', 'processed_monsoon_obs')
DIR_non_monsoon_gcm = config.get('Paths', 'processed_non_monsoon_gcm')
DIR_non_monsoon_observed = config.get('Paths', 'processed_non_monsoon_obs')

min_train_year = int(config.get('DataOptions', 'min_train_year'))
max_train_year = int(config.get('DataOptions', 'max_train_year'))
min_test_year = int(config.get('DataOptions', 'min_test_year'))
max_test_year = int(config.get('DataOptions', 'max_test_year'))
projection_dimensions = config.get('DataOptions', 'projection_dimensions')
channels = config.get('DataOptions', 'channels')

convlstm_kernels = config.get('ModelParams', 'convlstm_kernels')
convlstm_kernels = get_list_from_config(convlstm_kernels)

convlstm_kernel_sizes = config.get('ModelParams', 'convlstm_kernel_sizes')
convlstm_kernel_sizes = get_list_from_config(convlstm_kernel_sizes)

sr_block_kernels = config.get('ModelParams', 'sr_block_kernels')
sr_block_kernels = get_list_from_config(sr_block_kernels)

sr_block_kernel_sizes = config.get('ModelParams', 'sr_block_kernel_sizes')
sr_block_kernel_sizes = get_list_from_config(sr_block_kernel_sizes)

sr_block_depth = int(config.get('ModelParams', 'sr_block_depth'))
learning_rate_init = float(config.get('ModelParams', 'learning_rate_init'))
learning_rate_update_factor = float(config.get('ModelParams', 'learning_rate_update_factor'))
learning_rate_update_step = float(config.get('ModelParams', 'learning_rate_update_step'))
learning_rate_patience = float(config.get('ModelParams', 'learning_rate_patience'))
minimum_learning_rate = float(config.get('ModelParams', 'minimum_learning_rate'))
training_iters = int(config.get('ModelParams', 'training_iters'))
batch_size = int(config.get('ModelParams', 'batch_size'))
timesteps = int(config.get('ModelParams', 'timesteps'))
std_dev_observed=[]

def load_dataset(model_type):
    if model_type == 'monsoon':
        X = np.load(DIR_monsoon_gcm + 'X_low.npy')
        Y = np.load(DIR_monsoon_observed+ 'Y_low.npy')
    else:
        X = np.load(DIR_non_monsoon_gcm + 'X_high.npy')
        Y = np.load(DIR_non_monsoon_observed + 'Y_high.npy')        
    return X,Y


def normalize(data):
    data = data - data.mean()
    data = data / data.std()
    return data


def set_data(X, Y,):
    X_normalized = np.zeros((channels, np.max(X.shape), projection_dimensions[0], projection_dimensions[1]))

    for i in range(7):
        X_normalized[i,] = normalize(X[i,]) 

    Y_normalized = normalize(Y)

    print("Mean of GCM Data: ",X[0,].mean())
    print("Variance of GCM Data: ",X[0,].std(),end="\n")

    print("Mean of Obseved Data: ",Y.mean())
    print("Variance of Obseved Data: ",Y.std(),end="\n")

    std_observed = Y.std()  

    X = X_normalized.transpose(1,2,3,0)
    Y = Y_normalized.reshape(-1,projection_dimensions[0], projection_dimensions[1], 1)
    # print("X Shape: ", X.shape)
    # print("Y Shape: ", Y.shape)
    std_dev_observed.append(std_observed)
    return X, Y, std_observed


def data_generator(X,Y):
    total_years = max_test_year - min_train_year + 1
    train_years = max_train_year - min_train_year + 1
    n_days = np.max(X.shape)

    train_days = int((n_days/total_years)*train_years)

    train_x, train_y = X[:train_days], Y[:train_days]

    test_x, test_y = X[train_days:], Y[train_days:]

    time_steps = timesteps
    batch_size = batch_size
    train_generator = prep.sequence.TimeseriesGenerator(train_x, train_y.reshape(-1, projection_dimensions[0], projection_dimensions[1], 1),length=time_steps, batch_size=batch_size)
    test_generator = prep.sequence.TimeseriesGenerator(test_x, test_y.reshape(-1, projection_dimensions[0], projection_dimensions[1], 1),length=time_steps, batch_size=batch_size)

    return train_generator, test_generator

    def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def actual_rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred - y_true)*std_dev_observed[0])))

def train(clstm_model, model_type, train_generator, test_generator):
    adam = tf.keras.optimizers.Adam(lr=learning_rate_init)

    clstm_model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=[root_mean_squared_error, actual_rmse_loss])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"norm_clstm_{model_type}_prec_weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"./Graphs/norm_csltm_india_{model_type}_prec_Graph", histogram_freq=0, write_graph=True, write_images=False)
    termnan = tf.keras.callbacks.TerminateOnNaN()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=learning_rate_update_factor, patience=learning_rate_patience, min_delta=learning_rate_update_step, min_lr=minimum_learning_rate, verbose=1)
    
    callbacks_list = [checkpoint,tensorboard, reduce_lr, termnan]

    history = clstm_model.fit_generator(train_generator, callbacks=callbacks_list, epochs=training_iters, validation_data=test_generator ,verbose=1)
    return history


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode [train/ infer]", type=str)
    parser.add_argument("--model_type [monsoon/ non-monsoon][default: non-monsoon]", type=str, default='non-monsoon')
    parser.add_argument("--batch_size [default: 15]", type=int, default=15)
    parser.add_argument("--use_gpu [default: false]", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        print("Using GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        print("Using CPU")

    model_type = args.model_type 

    X,Y = load_dataset(model_type)
    X, Y, std_observed = set_data(X,Y)
    train_generator, test_generator = data_generator(X, Y)
    
    Aug_ConvLSTM_model = AugementedConvLSTM
    model = Aug_ConvLSTM_model.model(convlstm_kernels, convlstm_kernel_sizes, sr_block_kernels, sr_block_kernel_sizes, sr_block_depth)
    
    history = train(model, model_type, train_generator, test_generator)
    model.save_weights(f"epoch_{training_iters}_clstm_{model_type}_prec_weights.h5")    