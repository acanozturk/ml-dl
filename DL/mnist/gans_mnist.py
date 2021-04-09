######################################################################################
from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
######################################################################################




######################################################################################
# LOAD & PRE-PROCESS
(x_train, y_train), (x_test, y_test)= mnist.load_data()

# Normalization
x_train = (x_train.astype(np.float32) - 127 ) / 255

# Reshape 3D => 2D
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
######################################################################################




######################################################################################
# GENERATOR MODEL
# gen_input => 512 => relu => 512 => relu => 1024 => relu => 784 => tanh => gen_output

def generator():
    gen = Sequential()
    gen.add(Dense(units = 512, input_dim = 100))
    gen.add(ReLU())
    
    gen.add(Dense(units = 512))
    gen.add(ReLU())
    
    gen.add(Dense(units = 1024))
    gen.add(ReLU())
    
    gen.add(Dense(units = 784, activation = "tanh"))
    
    gen.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.0001, beta_1 = 0.5))
    
    return gen
######################################################################################
# GENERATOR
g = generator()
g.summary()
######################################################################################




######################################################################################
# DISCRIMINATOR MODEL
# disc_input => 1024 => relu => dropout => 512 => relu => dropout => 256 => relu => 1 => sigmoid => disc_output

def discriminator():
    disc = Sequential()
    disc.add(Dense(units = 1024, input_dim = 784))
    disc.add(ReLU())
    disc.add(Dropout(0.4))
    
    disc.add(Dense(units = 512))
    disc.add(ReLU())
    disc.add(Dropout(0.4))
    
    disc.add(Dense(units = 256))
    disc.add(ReLU())
    
    disc.add(Dense(units = 1, activation = "sigmoid"))
    
    disc.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.0001, beta_1 = 0.5))
    
    return disc
######################################################################################
# DISCRIMINATOR
d = discriminator()
d.summary()
######################################################################################




######################################################################################
# GAN MODEL
# gan_input => generator() => discriminator() => gan_output

def gan(disc, gen):
    disc.trainable = False
    
    gan_input = Input(shape = (100,))
    x = gen(gan_input)
    
    gan_output = disc(x)
    
    gan = Model(inputs = gan_input, outputs = gan_output)

    gan.compile(loss = "binary_crossentropy", optimizer = "adam")
    
    return gan
######################################################################################
# GAN
g = gan(d, g)
g.summary()
######################################################################################




######################################################################################
# GAN TRAIN
epochs = 10
batch_size = 256

for e in range(epochs):
    for _ in range(batch_size):
        
        noise = np.random.normal(0, 1 , [batch_size, 100])
        
        generated_image = g.predict(noise)
        
        image_batch = x_train[np.random.randint(low = 0, high = x_train.shape[0], size = batch_size)]
        
        x = np.concatenate([image_batch, generated_image], axis = 1)
      
        y_disc = np.zeros(batch_size * 2)
        y_disc[:batch_size] = 1
        
        d.trainable = True
        
        d.train_on_batch(x, y_disc)
        
        noise = np.random.normal(0, 1 , [batch_size, 100])
        
        y_gen = np.ones(batch_size)
        
        d.trainable = False
        
        gan.train_on_batch(noise, y_gen)

    print("Epoch:",e)                
######################################################################################


























