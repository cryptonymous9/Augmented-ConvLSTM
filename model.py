import os
import numpy as np

import tensorflow as tf
from tensorflow import keras 

class AugementedConvLSTM:
    
    """
    Implementaion of Augemnted ConvLSTM consisting of ConvLSTM layers and SR-Block

    """
    def __init__(self, channels=7, projection_height=129, projection_width=135, timesteps=5):
        """
        Initialize:

            channels: no. of climate variables involved (default:  7)

            projection_height: Height of the interpolated ESM projection (default for India:  129)
            
            projection_width: Width of the interpolated ESM projection (default for India:  135)
            
            timesteps: Number of consecutive days allowed for generating a single projections. 

        """
        
        self.channels = channels
        self.projection_height = projection_height
        self.projection_width = projection_width
        self.timesteps = timesteps
        
    def SR_block(self, x_in, sr_block_kernels=[128,64,1], sr_block_kernel_sizes=[9, 3, 5]):
        """
        Super-Resolution Block
        
            x_in: Input Image
            
            sr_block_kernels: (List)No. of Kernels for each Conv2D layer
                     Default: [128, 64, 1]
            
            sr_block_kernel_sizes: (List)Kernel Sized for each Conv2D layer
                     Default: [9, 5, 3]
        """
        
        x = keras.layers.Conv2D(filters = sr_block_kernels[0], kernel_size = sr_block_kernel_sizes[0], 
                                                padding='same', activation='relu')(x_in)
        
        x = keras.layers.Conv2D(filters = sr_block_kernels[1], kernel_size = sr_block_kernel_sizes[1], 
                                                padding='same', activation='relu')(x)
        
        x = keras.layers.Conv2D(filters = sr_block_kernels[2], kernel_size = sr_block_kernel_sizes[2], 
                                                padding='same', activation='relu')(x)
        output = keras.layers.Add()([x_in, x])
        
        return output

    def model(self, convlstm_kernels=[32,16,16], convlstm_kernel_sizes=[9,5,3], 
              sr_block_kernels=[128,64,1], sr_block_kernel_sizes=[9,3,5],  sr_block_depth=2):
        """
        Main Model:
        
        Input Dimension: (timesteps, projection_height, projection_width, channels)
        
            convlstm_kernels: (List length 3) Number of kernels for each ConvLSTM layer
                    Default:  [32,16,16]

            convlstm_kernel_sizes: (List length 3) Kernel sizes for each ConvLSTM layer
                    Default:  [9,5,3]
            
            sr_block_kernels: (List length 3) Number of kernels for each Conv2D layer in SR-Block
                    Default:  [129,64,1]
            
            sr_block_kernel_sizes: (List length 3) Kernel sizes for each Conv2D layer in SR-Block
                    Default:  [9,5,3]

            sr_block_depth: (int) No. of SR-blocks        
                    Default:  2

        Output Dimension: (projection_height, projection_width)

        """
        
        x_in = keras.layers.Input(shape = (self.timesteps, self.projection_height, 
                                                    self.projection_width, self.channels))
        
        x = keras.layers.ConvLSTM2D(filters = convlstm_kernels[0], kernel_size = convlstm_kernel_sizes[0], 
                                                    padding='same', return_sequences = True)(x_in)
        
        x = keras.layers.BatchNormalization()(x)

        x = b = keras.layers.ConvLSTM2D(filters = convlstm_kernels[1], kernel_size = convlstm_kernel_sizes[1], 
                                                    padding='same', return_sequences = True)(x)
        
        x = keras.layers.BatchNormalization()(x)

        x = b = keras.layers.ConvLSTM2D(filters = convlstm_kernels[2], kernel_size = convlstm_kernel_sizes[2], 
                                                    padding = 'same', return_sequences = False)(x)

        for i in range(sr_block_depth):
            b = self.SR_block(b)

        b = keras.layers.Conv2D(filters = convlstm_kernels[-1], kernel_size = convlstm_kernel_sizes[-1], 
                                                    padding = 'same', activation = 'relu')(b)

        x = keras.layers.Add()([x, b])

        x = keras.layers.Conv2D(filters = 1, kernel_size = convlstm_kernel_sizes[-2], padding='same')(x)

        return keras.models.Model(x_in, x)

        return model