# Libraries
from keras.models import Sequential # Layerları üzerine koyacağımız sıralı yapı, temel gibi düşünülebilir
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout # Kullanılacak CNN layerları
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt 
from glob import glob # Belirtilen dosyaları bulmaya yarar

# Datasetlerin pathleri
train_path = "C:/Users/Alican/Documents/Dersler/DL/fruits/Training/"
test_path = "C:/Users/Alican/Documents/Dersler/DL/fruits/Test/"

# Resimlerin boyutunu almak için
img = load_img(train_path + "Apple Braeburn/0_100.jpg")
x = img_to_array(img)


className = glob(train_path + "/*" ) # Classları /* ile baştan sona okur ve className arrayine yazar. len
numberOfClass = len(className) # Class sayısını verir
print("Number of Classes:", numberOfClass)

# CNN Modelini oluşturma, modelde 3 conv layerı olacak
model = Sequential() # Sıralı yapı olacağını belirttik

model.add(Conv2D(32, (3,3), input_shape = x.shape )) # Resimler 2 boyutlu olduğu için conv2d. Parantez içindeki değerler(Filtre(Feature) sayısı, konv. kernel boyutu, keras için input shape )
model.add(Activation("relu")) # Aktivasyon fonksiyonumuz relu olacak (x=<0 ise 0, değilse x)
model.add(MaxPooling2D()) # Pooling layerı


model.add(Conv2D(32, (3,3))) # Input shape sadece ilk conv layerı için gerekli
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Conv2D(64, (3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D())

# Fully conneceted layerı
model.add(Flatten())
model.add(Dense(1024)) # Dense layerındaki nöron sayısı
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Output layerı
model.add(Dense(numberOfClass)) # Output layerında nöron sayısı class sayısına eşit olmalı
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"]) # Softmax kullandığımız için lossumuz bu. 

batch_size = 32
epochs = 50

# Data Generation
'''
Datasetimizideki resim sayıları yetersiz olduğu için bu resimleri çeşitli şekilde yeniden generate ederek çeşitlilik artırılır
rescale: Normalizasyon. RGB resimleri grayscale yapar.
shear_range: Resmi rastgele şekilde çevirir
horizontal_flip: Yatay olarak çevirir
zoom: Zoom yapar
'''
train_data_generator = ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.3,
                                          horizontal_flip = True,
                                          zoom_range = 0.3) 

test_data_generator = ImageDataGenerator(rescale = 1./255) # Test datasıda normalize edilir

'''
flow_from_directory: Resimleri verilen path'den okur ve belirlenen generate özelliklerine göre generate eder
Resimler 100,100,3 boyutundadır. son index resmin tipini belirttiği için almaya gerek yok. boyutu [:2] ile okursak 100x100 elde ederiz
color_mode: Resimlerin tipi
class_mode : Tek class mı çok class mı var
'''
train_generator = train_data_generator.flow_from_directory(train_path, 
                                                           target_size=x.shape[:2],
                                                           batch_size = batch_size,
                                                           color_mode = "rgb",
                                                           class_mode = "categorical")

test_generator = test_data_generator.flow_from_directory(test_path, 
                                                         target_size=x.shape[:2],
                                                         batch_size = batch_size,
                                                         color_mode = "rgb",
                                                         class_mode = "categorical")

# Model train edilir
'''
steps_per_epoch: 1 epochta yapılması gereken batch sayısı. Normalde 400 adet resim var fakat 
biz 1600 resimle eğitmek istiyoruz bu resimler image generatordan geliyor.

validation_steps: Step per epochla aynı mantık
'''
history = model.fit_generator(generator = train_generator,
                    steps_per_epoch = 1600 // batch_size,
                    epochs = epochs,
                    validation_data = test_generator,
                    validation_steps = 800 // batch_size)

# Kayıt ve plot
model.save_weights("50epoch.h5")

print(history.history.keys())


plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()

'''
import json
with open("deneme.json","w") as f:
    json.dump(history.history, f)

# Kayıtlı weightleri import etme

import codecs
with codecs.open("buraya dosya adı yazılır", "r", encoding = "utf-8") as f:
    h = json.loads(f.read())
'''


























                                    



