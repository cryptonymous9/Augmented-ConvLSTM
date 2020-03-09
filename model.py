import os
import numpy as np
import tensorflow as tf

class AugementedConvLSTM():
    def __init__(self,channels=7, projection_height=129, projection_width=135, timesteps=5):
        self.channels = channels
        self.projection_height = projection_height
        self.projection_width = projection_width
        self.timesteps = timesteps
        
    def SR_block(self, x_in, sr_block_kernels, sr_block_kernel_sizes):
        x = tf.keras.layers.Conv2D(filters=sr_block_kernels[0], kernel_size=sr_block_kernel_sizes[0], padding='same')(x_in)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=sr_block_kernels[1], kernel_size=sr_block_kernel_sizes[1], padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=sr_block_kernels[2], kernel_size=sr_block_kernel_sizes[2], padding='same')(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        output = tf.keras.layers.Add()([x_in, x])
        return output

    def model(self, convlstm_kernels, convlstm_kernel_sizes, sr_block_kernels, sr_block_kernel_sizes,  sr_block_depth=2):
        x_in = tf.keras.layers.Input(shape=(self.timesteps, self.projection_height, self.projection_width, self.channels))
        x = tf.keras.layers.ConvLSTM2D(filters=convlstm_kernels[0], kernel_size=convlstm_kernel_sizes[0], padding='same', return_sequences=True)(x_in)
        x = tf.keras.layers.Activation(tf.keras.activations.tanh)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(filters=convlstm_kernels[1], kernel_size=convlstm_kernel_sizes[1], padding='same', return_sequences=True)(x)
        x = tf.keras.layers.Activation(tf.keras.activations.tanh)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(filters=convlstm_kernels[2], kernel_size=convlstm_kernel_sizes[2], padding='same',return_sequences=False)(x)
        x = b = tf.keras.layers.BatchNormalization()(x)
        
        for i in range(sr_block_depth):
            b = self.SR_block(b, sr_block_kernels, sr_block_kernel_sizes)
        
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(b)
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=(5,5), padding='same')(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        model = tf.keras.models.Model(x_in, x) 

        return model
