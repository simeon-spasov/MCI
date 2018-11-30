from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Reshape, Dense, ELU, concatenate, add, Lambda, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.metrics import binary_crossentropy
import numpy as np
import math

import sys
#sys.path.append('/home/ses88/venv/PropagAgeing')
sys.path.append('/Users/simeonspasov/DL files/MCI advanced')
from sepconv3D import SeparableConv3D
from augmentation import CustomIterator


class Parameters():
    def __init__ (self, param_dict):
        self.w_regularizer = param_dict['w_regularizer']
        self.batch_size = param_dict['batch_size']
        self.drop_rate = param_dict['drop_rate']
        self.epochs = param_dict['epochs']
        self.gpu = param_dict['gpu']
        self.model_filepath = param_dict['model_filepath'] + '/net.h5'
        self.num_clinical = param_dict['num_clinical']
        self.image_shape = param_dict['image_shape']        
        
class Net ():
    def __init__ (self, params):
        self.params = params
        self.xls = Input (shape = (self.params.num_clinical,))
        self.mri = Input (shape = (self.params.image_shape))
        self.jac = Input (shape = (self.params.image_shape))
        
        xalex3D = XAlex3D(w_regularizer = self.params.w_regularizer, drop_rate = self.params.drop_rate)
    
        with tf.device(self.params.gpu):
            self.fc_mci = xalex3D (self.mri, self.jac, self.xls) 
            self.output_mci = Dense(units = 1, activation = 'sigmoid', name = 'mci_output') (self.fc_mci)
        
    def train (self, data):
        train_data, val_data = data
        train_samples = train_data[0].shape[0]
        val_samples = val_data[0].shape[0]
        
        data_flow_train = CustomIterator (train_data, batch_size = self.params.batch_size,
                                          shuffle = True)
        data_flow_val = CustomIterator (val_data, batch_size = self.params.batch_size,
                                          shuffle = True)


        self.model = Model(inputs = [self.mri, self.jac, self.xls], outputs = self.output_mci)        
        lrate = LearningRateScheduler(step_decay)    
        callback = [lrate]    
        optimizer = Adam(lr=0e-3) 
        self.model.compile(optimizer = optimizer, loss = [binary_crossentropy], metrics =['acc'])  
        #This version implements pMCI/sMCI learning only
        
        history = self.model.fit_generator (data_flow_train,
                   steps_per_epoch = train_samples/self.params.batch_size,
                   epochs = self.params.epochs,
                   callbacks = callback,
                   shuffle = True,
                   validation_data = data_flow_val,
                   validation_steps =  val_samples/self.params.batch_size)
        
        return history.history

    def predict (self, data_test):
        test_mri, test_jac, test_xls, test_labels = data_test
        preds = self.model.predict ([test_mri, test_jac, test_xls])
        return preds

    def evaluate (self, data_test):
        test_mri, test_jac, test_xls, test_labels = data_test
        metrics = self.model.evaluate (x = [test_mri, test_jac, test_xls], y = test_labels, batch_size = self.params.batch_size)
        return metrics



def XAlex3D(w_regularizer = None, drop_rate = 0.) :
  
    #3D Multi-modal deep learning neural network (refer to fig. 4 for chain graph of architecture)
    def f(mri_volume, mri_volume_jacobian, clinical_inputs):
    
        #First conv layers
        conv1_left = _conv_bn_relu_pool_drop(24, 11, 13, 11, strides = (4, 4, 4), w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (mri_volume)

        conv1_right = _conv_bn_relu_pool_drop(24, 11, 13,  11, strides = (4, 4, 4), w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (mri_volume_jacobian)
    
        #Second layer
        conv2_left =_conv_bn_relu_pool_drop(48, 5, 6, 5, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv1_left)
    
        conv2_right =_conv_bn_relu_pool_drop(48, 5, 6, 5, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv1_right)
    
        conv2_concat = concatenate([conv2_left, conv2_right], axis = -1)
    
        #Introduce Middle Flow (separable convolutions with a residual connection)
        conv_mid_1 = mid_flow (conv2_concat, drop_rate, w_regularizer, filters = 96)
    
        #Split channels for grouped-style convolution
        conv_mid_1_1= Lambda (lambda x:x[:,:,:,:,:48]) ( conv_mid_1 )
        conv_mid_1_2 = Lambda (lambda x:x[:,:,:,:,48:]) (conv_mid_1 )
        
        conv5_left = _conv_bn_relu_pool_drop (24, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv_mid_1_1)
    
        conv5_right = _conv_bn_relu_pool_drop (24, 3, 4, 3, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (conv_mid_1_2)
    
        conv6_left = _conv_bn_relu_pool_drop (8, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv5_left)

        conv6_right = _conv_bn_relu_pool_drop (8, 3, 4, 3, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=True) (conv5_right)
        
        conv6_concat = concatenate([conv6_left, conv6_right], axis = -1)
    
    
        #Flatten 3D conv network representations
        flat_conv_6 = Reshape((np.prod(K.int_shape(conv6_concat)[1:]),))(conv6_concat)
    

        #2-layer Dense network for clinical features
        vol_fc1 = _fc_bn_relu_drop(32,  w_regularizer = w_regularizer,
                               drop_rate = drop_rate)(clinical_inputs)

        flat_volume = _fc_bn_relu_drop(10, w_regularizer = w_regularizer,
                                   drop_rate = drop_rate)(vol_fc1)   
    
        #Combine image and clinical features embeddings
    
        fc1 = _fc_bn_relu_drop (10, w_regularizer, drop_rate = drop_rate) (flat_conv_6)
        flat = concatenate([fc1, flat_volume])
    
        #Final 4D embedding
        fc2 = _fc_bn_relu_drop (4, w_regularizer, drop_rate = drop_rate) (flat) 


        return fc2
    return f



def _fc_bn_relu_drop (units, w_regularizer = None, drop_rate = 0., name = None):
    #Defines Fully connected block (see fig. 3 in paper)
    def f(input):  
        fc = Dense(units = units, activation = 'linear', kernel_regularizer=w_regularizer, name = name) (input) #was 2048 initially
        fc = BatchNormalization()(fc)
        fc = ELU()(fc)
        fc = Dropout (drop_rate) (fc)
        return fc
    return f

def _conv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding = 'same', w_regularizer = None, 
                            drop_rate = None, name = None, pool = False):
   #Defines convolutional block (see fig. 3 in paper)
   def f(input):
       conv = Conv3D(filters, (height, width, depth),
                             strides = strides, kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
       norm = BatchNormalization()(conv)
       elu = ELU()(norm)
       if pool == True:       
           elu = MaxPooling3D(pool_size=3, strides=2) (elu)
       return Dropout(drop_rate) (elu)
   return f

def _sepconv_bn_relu_pool_drop (filters, height, width, depth, strides = (1, 1, 1), padding = 'same', depth_multiplier = 1, w_regularizer = None, 
                            drop_rate = None, name = None, pool = False):
    #Defines separable convolutional block (see fig. 3 in paper)
    def f (input):
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                             strides = strides, depth_multiplier = depth_multiplier,kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
        sep_conv = BatchNormalization()(sep_conv)
        elu = ELU()(sep_conv)
        if pool == True:       
           elu = MaxPooling2D(pool_size=3, strides=2, padding = 'same') (elu)
        return Dropout(drop_rate) (elu)
    return f


def mid_flow (x, drop_rate, w_regularizer, filters = 96):
    #3 consecutive separable blocks with a residual connection (refer to fig. 4)
    residual = x   
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _sepconv_bn_relu_pool_drop (filters, 3, 3, 3, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
    x = add([x, residual])
    return x

def step_decay (epoch):
    #Decaying learning rate function
    initial_lrate = 1e-3                                                                              
    drop = 0.3                                                                                        
    epochs_drop = 10.0                                                                                
    lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))                                    
    return lrate    

