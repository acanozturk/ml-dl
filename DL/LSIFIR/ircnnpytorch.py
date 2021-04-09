#######################################################################################
import torch
import torch.nn as nn
import torch.utils.data 
import torch.optim as optim
import torch.nn.functional as F # Fonksiyonları barıdıran kütüphane(Avtivation,relu vs.)
from PIL import Image # Resimlere pre-process yaparken kullanacağız
import numpy as np
import os # Resimleri import ederken kullanacağız
import time # Run-time görmek için kullanacağız
'''
Pytorch ile gpu'da(CUDA gerekir) çalışabiliriz. Bu yöntem modeli daha hızlı eğitir.
'''
#######################################################################################




#######################################################################################
# DATASET LOAD AND PRE-PROCESS #
'''
1- read_images: path ve kaç adet resim okunacağını alır.
2- 0'lardan oluşan array yapar. Bu arrayin boyutu resim adeti(num_img) ve resimlerin boyutuna(64*32) bağlıdır.
3- os.listdir(): içine yazılan konuma gider.
4- img_path: Resimlerin bulunduğu klasöre gider. Daha sonra bu klasör içindeki resimleri sırayla okur.
5- asarray: Resimleri uint8'e çevirmek için kullandık.
6- Resimleri array'de depolayacağımız için vektöre çevirmek gerekir. Bunun için flatten yapıyoruz.
7- Son olarak i. indeksteki resimden başlar ve resimler bitene kadar bunu data değişkenine yazdırır. i'yi 1 arttırır.
'''
def read_images(path, num_img):
    array = np.zeros([num_img, 64*32])
    
    i = 0   
    for img in os.listdir(path):
        
        img_path = path + "\\" + img
        img = Image.open(img_path, mode = "r")
        
        data = np.asarray(img, dtype = "uint8")
        data = data.flatten()
        
        array[i,:] = data
        i += 1
 
    return array
#######################################################################################
    



#######################################################################################
# PATH GÖSTERME #
'''
1- Önce resimlrin pathlerini verdik
2- Sayıları dökümantasyondan biliyoruz
3- Path ve resim sayısı belirttikten sonra pre-process metoduna yolladık.
4- Pre-process'ten gelen arrayleri tensöre çevirdik. Yani numpy arrayleri pytorch arraye çevirdik.
5- Burada x tensörü resimleri, y tensörü labelları temsil etmektedir. y labelı negatifler için 0 pozitifler için 1.
'''

# Train    
train_neg_path = r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Train\neg"
train_neg_num_img = 43390 
train_neg_array = read_images(train_neg_path, train_neg_num_img)
x_train_neg_tensor = torch.from_numpy(train_neg_array)
y_train_neg_tensor = torch.zeros(train_neg_num_img, dtype = torch.long)

train_pos_path = r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Train\pos"
train_pos_num_img = 10208
train_pos_array = read_images(train_pos_path, train_pos_num_img)
x_train_pos_tensor = torch.from_numpy(train_pos_array)
y_train_pos_tensor = torch.ones(train_pos_num_img, dtype = torch.long)


# Test
test_neg_path =  r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Test\neg"
test_neg_num_img = 22050
test_neg_array = read_images(test_neg_path, test_neg_num_img)
x_test_neg_tensor = torch.from_numpy(test_neg_array[:20855,:])
y_test_neg_tensor = torch.zeros(20855, dtype = torch.long)

test_pos_path =  r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Test\pos"
test_pos_num_img = 5944
test_pos_array = read_images(test_pos_path, test_pos_num_img)
x_test_pos_tensor = torch.from_numpy(test_pos_array)
y_test_pos_tensor = torch.ones(test_pos_num_img, dtype = torch.long)


# Boyutları yazdırma
print("x_train_neg shape:", x_train_neg_tensor.size())
print("y_train_neg shape:", y_train_neg_tensor.size())

print("x_train_pos shape:", x_train_pos_tensor.size())
print("y_train_pos shape:", y_train_pos_tensor.size())

print("x_test_neg shape:", x_test_neg_tensor.size())
print("y_test_neg shape:", y_test_neg_tensor.size())

print("x_test_pos shape:", x_test_pos_tensor.size())
print("y_test_pos shape:", y_test_pos_tensor.size())
#######################################################################################




#######################################################################################
# CONCATENATION
'''
1- Pozitif ve negatif resimleri tek bir değişkene toplamak için yapıyoruz
2- Sonundaki değer 0 olursa satırları(yukarıdan aşağı), 1 olursa sütunları birleştir. Biz satırları birleştireceğiz
'''

# Train
x_train = torch.cat((x_train_neg_tensor, x_train_pos_tensor), 0)
y_train = torch.cat((y_train_neg_tensor, y_train_pos_tensor), 0)

# Test
x_test = torch.cat((x_test_neg_tensor, x_test_pos_tensor), 0)
y_test = torch.cat((y_test_neg_tensor, y_test_pos_tensor), 0)

# Boyutları yazdırma
print("x_train size:", x_train.size())
print("y_train size:", y_train.size())

print("x_test size:", x_test.size())
print("y_test size:", y_test.size())
#######################################################################################




#######################################################################################
# Hyperparameters
num_epochs = 20
batch_size = 8933
learning_rate = 0.00001
num_of_classes = 2  
#######################################################################################




#######################################################################################
# TRAIN VE TEST DATALARINI BİRLEŞTİRME. pyrotch formatına uydurmak için

train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
#######################################################################################




#######################################################################################
# CNN CLASS OLUŞTURMA
'''
1- Pytorch kütüphanesinin NN modellerini(Conv2d,MaxPooling vs.) kullanacağız. Yani inherit edeceğiz.
2- def __init__(self): Bu metodun içinde kullanacağımız modelleri tanımlayacağız.

    super(Net, self).__init__(): Inheritance gerçekleştirmeye yarar.
    self.conv1 = nn.Conv2d(input_image_channel, output_channel  , kernel_size)
    self.pool = nn.MaxPool2d(kernel_size)
    self.fc1 = nn.Linear(, )
    self.fc2 = nn.Linear(, )
    self.fc3 = nn.Linear(, )

3- def forward(self, x): Bu metodun içinde, init metodunda belirttiğimiz modelleri sıraya koyacağız.
    
     F: pytorch fonksiyonları böyle import ettik.
     
     input(x) => Conv1 => Relu => MaxPooling => Conv2 => Relu => MaxPooling => Flatten => FC1 => Relu => FC2 => Relu => FC3
     
     x = self.pool(F.relu(self.conv1(x)))
     x = self.pool(F.relu(self.conv2(x)))
        
     x = x.view(-1, 16*13*5): Pytorch flatten böyle yapılır.
        
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = self.fc3(x)
'''

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        
        self.fc1 = nn.Linear(16*13*5, 520)
        self.fc2 = nn.Linear(520, 130)
        self.fc3 = nn.Linear(130, num_of_classes)
        
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16*13*5)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
#######################################################################################
        
    
    
    
#######################################################################################
# MODEL EĞİTİMİ PRE-PROCESS

# Model fit etme
net = Net()

# Loss fonksiyonu     
loss_function = nn.CrossEntropyLoss()   

# Optimizasyon 
optimizer = optim.SGD(net.parameters(), # Weight ve biasları parametre olarak alır
                      lr = learning_rate,
                      momentum = 0.8)  # SGD hızlanma parametresi        
#######################################################################################




#######################################################################################
# MODEL EĞİTİMİ
'''
1- start,end = time.time(): Run time hesaplamak için 
2- Accuracy ve lossları depolamak için listeleri oluştuduk
3- For döngüsü epoch sayısı kadar dönecek. Her epochta
    for i, data in enumerate(train_loader, 0): Trainloader ve 0'ı birleştirerek index ve datayı döndürür
        inputs, labels = data: Datanın içinde resimler ve labelları var
        inputs = inputs.view(batch_size, 1, 64, 32): pytorch için resize yapar
        inputs = inputs.float(): inputları floata çevirir
        optimizer.zero_grad(): Her adımda gradları 0'lar. pytorch otomatik yapmıyor
        outputs = net(inputs): Forward propagation
        loss = loss_function(outputs, labels): Loss bulma
        loss.backward(): Backward propagation
        optimizer.step(): Parametre update
4- Pytorchh accuracyi göstermiyor kendimiz hesaplamak zorundayız.
'''

start = time.time()

train_acc = []
test_acc = []
loss_list = []


for epochs in range(num_epochs):
    for i,data in enumerate(train_loader, 0):
        
        inputs, labels = data
        
        inputs = inputs.view(batch_size, 1, 64, 32)
        inputs = inputs.float()
        
        
        optimizer.zero_grad()
    
        outputs = net(inputs)
    
        loss = loss_function(outputs, labels)
        
        loss.backward()
    
        optimizer.step()
    
# Train Accuracy
   
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in train_loader:
            
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
                
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
    
    
    train_accuracy = 100*correct/total
    print("Train accuracy:", train_accuracy)
    train_acc.append(train_accuracy)
    
    
# Test Accuracy
   
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
                
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
    
    
    test_accuracy = 100 *correct/total
    print("Test accuracy:", test_accuracy)
    test_acc.append(test_accuracy)


print("Training done!")

end = time.time()

process_time = (end - start) / 60  
print("Process time: %.1f minutes"%process_time)    
#######################################################################################
       
        
        
    
















































