#####################################################################################
from keras.applications.vgg16 import VGG16
from keras.models import Sequential 
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt 
from glob import glob
#####################################################################################




#####################################################################################
train_path = "C:/Users/Alican/Documents/Dersler/DL/fruits/Training/"
test_path = "C:/Users/Alican/Documents/Dersler/DL/fruits/Test/"
img = load_img(train_path + "Apple Braeburn/0_100.jpg")
x = img_to_array(img)
print("Image shape:",x.shape)
#####################################################################################




#####################################################################################
className = glob(train_path + "/*" )
numberOfClass = len(className) 
print("Number of Classes:", numberOfClass)
#####################################################################################




#####################################################################################
# VGG16 MODEL
'''
1.VGG Modeli 1000 classa göre eğitilmiş fakat bizim datasetimiz 120 classtan oluşuyor.
 Bu sebeple default output silip kendi layerımzı ekleyeceğiz
 
2. Burada kendi modelimiz oluşturduk. Modelin içine vgg layerlarını(son layer hariç) koyduk.
model = Sequential()
for i in range(len(vgg_layerlist)-1):
    model.add(vgg_layerlist[i])  
    
3. Burada vgg layerlarını tekrar eğitmeyeceğimizi belirttik.
for layers in model.layers:
    layers.trainable = False
    
4. Kendi dense layerımızı oluşturduk ve modele ekledik.
'''
vgg = VGG16()

#print(vgg.summary())
vgg_layerlist = vgg.layers
#print(vgg_layerlist)

model = Sequential()
for i in range(len(vgg_layerlist)-1):
    model.add(vgg_layerlist[i])

#print(model.summary())

for layers in model.layers:
    layers.trainable = False
    
model.add(Dense(numberOfClass, activation = "softmax"))
#print(model.summary())

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
#####################################################################################




#####################################################################################
# TRAIN-TEST
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224)) # VGG input resimleri 2224x224 olduğu için reshape yaptık
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224))
#####################################################################################




#####################################################################################
# Hyperparameters
batch_size = 32
epochs = 10
#####################################################################################




#####################################################################################
# MODEL TRAIN
history = model.fit_generator(train_data,
                              steps_per_epoch = 1600 // batch_size,
                              epochs = epochs,
                              validation_data = test_data,
                              validation_steps = 800 // batch_size)
#####################################################################################


















#####################################################################################


