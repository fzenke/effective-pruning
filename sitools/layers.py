from keras.layers import Dense, Wrapper
import keras.backend as K
import tensorflow as tf
import numpy as np


class PruneConnect(Wrapper):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(PruneConnect, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            # Copy unpruned parameters
            self.kernel = self.layer.kernel
            self.bias = self.layer.bias
            # Prepare all ones masks
            self.kernel_mask = K.variable(np.ones(self.kernel.shape))
            self.bias_mask = K.variable(np.ones(self.bias.shape))
        super(PruneConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    def set_kernel_mask(self, mask):
        sess = K.get_session()
        self.kernel_mask.assign(K.cast(mask, self.kernel.dtype)).eval(session=sess)
        
    def set_bias_mask(self, mask):
        sess = K.get_session()
        self.bias_mask.assign(K.cast(mask, self.bias.dtype)).eval(session=sess)
         
    def get_nb_parameters(self):
        return K.sum([K.prod(self.kernel_mask.shape), K.prod(self.bias_mask.shape)])
    
    def get_nz_parameters(self):
        return K.sum(K.concatenate([K.flatten(self.kernel_mask), K.flatten(self.bias_mask)]))
    
    def get_sparseness(self):
        return K.mean(K.concatenate([K.flatten(self.kernel_mask), K.flatten(self.bias_mask)]))

    def call(self, x):
        self.layer.kernel = self.kernel*self.kernel_mask
        self.layer.bias   = self.bias*self.bias_mask
        return self.layer.call(x)

