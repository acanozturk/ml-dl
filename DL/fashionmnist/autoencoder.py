from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json, codecs
import warnings
warnings.filterwarnings("ignore")

#############################################################################
# LOAD & PRE-PROCESS
'''
1. Autoencoders unsupervised learning olduğu için labellara(y) gerek yok. _ koyarsak o değeri almaz
2. x_train'i uint8'den floata çevirdik.
3. Train ve test datalarını 2dye çevirdik
'''
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Reshape & Normalization
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3D => 2D
x_train = x_train.reshape((len(x_train), x_train.shape[1:][0] * x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test), x_test.shape[1:][0] * x_test.shape[1:][1]))
#############################################################################   




#############################################################################   
# ENCODER
input_img = Input(shape = (784,)) # Input layer

enc = Dense(32, activation = "relu")(input_img) # Input ve dense layerı bağladık
enc = Dense(16, activation = "relu")(enc) # Layerları bağladık

# DECODER
dec = Dense(32, activation = "relu")(enc)
dec = Dense(784, activation = "sigmoid")(dec)

# AUTOENCODER
autoencoder = Model(input_img, dec)
autoencoder.compile(optimizer = "rmsprop", loss = "binary_crossentropy")

history = autoencoder.fit(x_train, x_train,  # Unsupervised learning olduğu için y_train yok x_train yazılır
                          epochs = 20,
                          batch_size = 256,
                          shuffle = True,
                          validation_data = (x_train, x_train))

autoencoder.save_weights("autoencoder.h5")
#############################################################################   





#############################################################################
# VISUALIZATION
plt.plot(history.history["loss"], label = "Train loss")
plt.plot(history.history["val_loss"], label = "Val loss")

plt.legend()
plt.show()
#############################################################################




#############################################################################
# SAVE-LOAD
with open("autoencoder.json","w") as f:
    json.dump(history.history, f)

with codecs.open("autoencoder.json","r", encoding = "utf-8") as f:
    n = json.loads(f.read())
#############################################################################





#############################################################################
encoder = Model(input_img, enc)
enc_img = encoder.predict(x_test)

plt.imshow(x_test[1500].reshape(28,28))
plt.show()

plt.imshow(enc_img[1500].reshape(4,4))
plt.show()

dec_img = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))

for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.axis("off")
    
    ax = plt.subplot(2,n,i+1)
    plt.imshow(dec_img[i].reshape(28,28))
    plt.axis("off")
plt.show()
#############################################################################   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























