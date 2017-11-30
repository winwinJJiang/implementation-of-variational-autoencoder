#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:48:43 2017

@author: jiangj1
"""


import tensorflow as tf
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras import backend as K

from keras.datasets import mnist

import matplotlib.pyplot as plt
import os
from PIL import Image
# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3
epochs = 1
batch_size=100
optimizer='rmsprop'
## start get the vae model



img_rows, img_cols, img_chns = 28, 28, 1
original_dim=28*28


latent_dim=2  # here set the latent space of z
filters = 64
# convolution kernel size
num_conv = 3
epsilon_std = 1.0
batch_size=100
intermediate_dim=128
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean - K.exp(z_log_var) * epsilon



x = Input((28,28,1))
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)


z_mean = Dense(latent_dim)(hidden)    # for the z mean
z_log_var = Dense(latent_dim)(hidden)   ## for the z_log_var, all can be learned by the NN end of encoder

decoder_upsample = Dense(filters * 14 * 14, activation='relu')
decoder_hid = Dense(intermediate_dim, activation='relu')


if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 14, 14)
else:
    output_shape = (batch_size, 14, 14, filters)


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])  ## sampling the z_mean and z_log_var  need to use the function
#epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
#z=z_mean + K.exp(z_log_var) * epsilon


decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)  ## reshape the shape of the tensor  14*14*batch size  clear!
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

model = Model(input=[x], output=[x_decoded_mean_squash])
encoder=Model(x,z_mean)

model.summary()


import sys
from keras import backend as K
import numpy as np
from keras import metrics

IntType = 'int32'
FloatType = 'float32'

#calculate the squared distance between x and y


class VAE_LOSS:

    
    def __init__(self,
                 x, # 
                 z_mean,
                 z_log_var,
                 x_decoder):

        self.x = x
        self.x_decoder = x_decoder
        self.z_mean=z_mean
        self.z_log_var=z_log_var
        

    def KL_loss(self,z_mean, z_log_var):
        kl_loss=-0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        return kl_loss
 

    def decoder_loss(self,X,Y):  ## the decoder loss

        #d_loss = -K.sum(K.log(K.square(X-Y)))     # mine    # this is wrong
        X=K.flatten(X)
        Y=K.flatten(Y)
        d_loss=28 * 28 * metrics.binary_crossentropy(X, Y)             # this is right                               # keras
        return d_loss

    def KerasVAECost(self, y_true,y_pred,x,  z_mean,z_logvar,x_decoder):
       
        x=self.x
        kl_loss=self.KL_loss(z_mean,z_logvar)

        de_loss=self.decoder_loss(x,x_decoder)      
          
        vae_loss =kl_loss+de_loss
        #vae_loss=kl_loss
        #vae_loss=-100
        return vae_loss
    
#model, x, z_mean, z_log_var, x_decoded_mean_squash,encoder= VAE_Model(img_rows)

print 'debuging'
print x.shape
print z_mean.shape
print z_log_var.shape
print x_decoded_mean_squash.shape

   
model.compile(loss=lambda y_true,y_pred:VAE_LOSS(x,z_mean,z_log_var,x_decoded_mean_squash).KerasVAECost(y_true,y_pred,x,z_mean,z_log_var,x_decoded_mean_squash), optimizer=optimizer, metrics=None)


## start load data
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
y_train = y_train.reshape((x_train.shape[0],) + (1,1,1))

print('x_train.shape:', x_train.shape)

print ('start fitting the model :OK')

model.fit(x_train,y_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)
        #validation_data=None,
        #validation_steps=None)


x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_test_encoded = encoder.predict(x_test, batch_size=1)

#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()
#plt.imsave('/lila/data/deasy/Eric_Data/self_implementation/VAE/VAE_Keras/result/z_distributation_'+'.png',cmap=plt.cm.gray)
        

decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

#plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
#plt.show()


output_dir='/lila/data/deasy/Eric_Data/self_implementation/VAE/VAE_Keras/save/'
figure_save = Image.fromarray(figure, mode='l')  # L specifies greyscale
outfile = os.path.join(output_dir, 'epoch__Train_{}.png'.format(epochs))
figure_save.save(outfile)

plt.savefig(output_dir, format='png', dpi=300)