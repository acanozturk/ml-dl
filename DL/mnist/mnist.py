# LIBRARIES
from keras.models import Sequential # Layerları üzerine koyacağımız sıralı yapı, temel gibi düşünülebilir
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization # Kullanılacak CNN layerları
from keras.utils import to_categorical # One-hot encoding, kerasın anlaması için
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# LOAD AND PRE-PROCESS
'''
1. csv dosyayı okuyup dataya atar
2. data.values ile arraye çevirir
3. shuffle ile dataların sırasını bozarak karıştırır
4. x(resimler) ve y(labellar) olmak üzere datayı böler ve keras için reshape eder
5. astype ile y(label) değerlerini integera çevirir
6. to_categorical ile labellara one-hot encoding yapar
7. x ve y değerlerini döndürür
'''
def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.values
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1, 28, 28, 1) / 255.0  
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes = len(set(y)))
    
    return x,y



train_path = "C:/Users/Alican/Documents/Dersler/DL/mnist/mnist_train.csv"
test_path = "C:/Users/Alican/Documents/Dersler/DL/mnist/mnist_test.csv"

x_train, y_train = load_and_preprocess(train_path)
x_test, y_test = load_and_preprocess(test_path)

print("x_train size:", x_train.shape)
print("y_train size:", y_train.shape)
print("x_test size:", x_test.shape)
print("y_test size:", y_test.shape)



# CNN MODELİ
numberofClass = y_train.shape[1] # Class sayımız y_trainin 1. indexi verir

model = Sequential()

# Input Conv Layer
model.add(Conv2D(input_shape = (28,28,1), filters = 16, kernel_size =(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu")) 
model.add(MaxPooling2D())

# Conv Layer 1
model.add(Conv2D(filters = 64, kernel_size =(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu")) 
model.add(MaxPooling2D())

# Conv Layer 2
model.add(Conv2D(filters = 64, kernel_size =(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu")) 
model.add(MaxPooling2D())

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))


# Output Layer
model.add(Dense(units = numberofClass)) 
model.add(Activation("softmax"))



model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"]) # Softmax kullandığımız için lossumuz bu. 

batch_size = 400
epochs = 50


# Model fit etme
hist = model.fit(x_train, y_train, 
                 validation_data = (x_test, y_test),
                 epochs = epochs, 
                 batch_size = batch_size)

model.save_weights("50epoch.h5")

print(hist.history.keys())

plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(hist.history["accuracy"], label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()















