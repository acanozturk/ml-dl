#######################################################################################
import torch
import torch.nn as nn
import torch.utils.data 
import torch.optim as optim
import torch.nn.functional as F # Fonksiyonları barıdıran kütüphane(Avtivation,relu vs.)
from PIL import Image # Resimlere pre-process yaparken kullanacağız
import numpy as np
import os # Resimleri import ederken kullanacağız
import time
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
x_train_neg_tensor = torch.from_numpy(train_neg_array[:42000,:])
y_train_neg_tensor = torch.zeros(42000, dtype = torch.long)

train_pos_path = r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Train\pos"
train_pos_num_img = 10208
train_pos_array = read_images(train_pos_path, train_pos_num_img)
x_train_pos_tensor = torch.from_numpy(train_pos_array[:10000,:])
y_train_pos_tensor = torch.ones(10000, dtype = torch.long)


# Test
test_neg_path =  r"C:\Users\Alican\Documents\Dersler\DL\LSIFIR\Classification\Test\neg"
test_neg_num_img = 22050
test_neg_array = read_images(test_neg_path, test_neg_num_img)
x_test_neg_tensor = torch.from_numpy(test_neg_array[:18056,:])
y_test_neg_tensor = torch.zeros(18056, dtype = torch.long)

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
batch_size = 2000
learning_rate = 0.00001
num_classes = 2  
#######################################################################################




#######################################################################################
# TRAIN VE TEST DATALARINI BİRLEŞTİRME. pyrotch formatına uydurmak için

train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
#######################################################################################




#######################################################################################
# RESNET CONV. CHANNELS
'''
1- def convnxn(input_channel(RGB=3,GRAYSCALE=1 vs.), Nöron sayısı, Kernel pixel atlama sayısı):nxn kernellı konvolüsyon
'''
def conv3x3(in_planes, out_planes, stride = 1):       
        
   return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False) 



def conv1x1(in_planes, out_planes, stride = 1):       
        
   return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False) 

#######################################################################################
   




#######################################################################################
# RESNET BASIC BLOCK
'''
1- class BasicBlock(nn.Module): pytorch neural network classını inherit eder
2- def __init__(): Temel yapıları burada oluştaracağız
3- def forward(): Temel yapıları burada bağlayacağız
4- Son relu adımında identity ve nn outputunu toplayacağız.
   x => conv1 => bn1 => relu => dropout => conv2 => bn2 => relu => output (Normal nn)
   x ====================================================> relu => output (Shortcut(Identity))
5- Örneğin en başta x boyutu 4x4 olsun. Layerlardan geçtiken sonra boyutu değişecektir. Fakat identity kısmında x orijinal boyutuyla
 reluya  gireceği için bu boyutlarını eşitlememiz lazım. Bunu da downsampling ile yaparız.
'''   
class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0.9)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    
    
    def forward(self,x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: # Eğer boyutları eşit değilse downsample ederek sizeları eşitler
            identity = self.downsample(x)
        
        out = out + identity
        
        out = self.relu(out)
        
        return out
#######################################################################################        
        
    
    
    
###################################################################################### 
# RESNET MODEL OLUŞTURMA
'''
1- def _make_layer(): Basic blockları birleşitirerek derin bir resnet yapısı oluşturcağımız yer

   def _make_layer(self, BASIC BLOCK, INPUT CHANNELARI, BASIC BLOCK SAYISI)
     downsample = None
     
  - Eğer stride 1 değilse veya input chanellarımızda genişleme varsa(expansion ile genişletebiliriz)   
  
     if stride != 1 or self.inplanes != planes*block.expansion:
         downsample = nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride),
                                    nn.BatchNorm2d(planes*block.expansion))
            
    layers = [] 
        
    layers.append(block(self.inplanes, planes, stride, downsample)): Buradaki block basicblock
    self.inplanes = planes*block.expansion: Blokları genişletmek istersek gerekli
        
    for _ in range(1, blocks): Bşlangıçta 1 blok eklediğimiz için 1'den başlattık blok sayısına kadar döndürdük.
        layers.append(block(self.inplanes, planes))
            
    return nn.Sequential(*layers)

2- x => conv1 => bn1 => relu => maxpool => makelayer1 => makelayer2 => makelayer3 => avgpool => flatten => fc => output
   
3- Resnet layerlarındaki weighleri init. edeceğiz. Burada self.modules içideki m parametresine sırayla conv1 bn1 vs. layerları yükler.
   Eğer m Conv layerındaysa weightleri kaiming ile 0a çok yakın olarak günceller. 
   Eğer BatchNorm layerındaysa weighleri 1 biası 0 yapar.
 
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
'''        
class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes = num_classes):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # Verdiğimiz output buyutuna göre pooling uygular. Burada 1x1 output için yapacak
        
        self.fc = nn.Linear(256*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
          
           
            
            
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride),
                                       nn.BatchNorm2d(planes*block.expansion))
            
        layers = []
        
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
            



    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.fc(x)
        
        return x

model = ResNet(BasicBlock, [2,2,2])
###################################################################################### 




###################################################################################### 
# Model fit etme       


loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
######################################################################################




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

total_step = len(train_loader)


for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        
        images, labels = data
        
        images = images.view(batch_size, 1, 64, 32)
        images = images.float()
        
        
        optimizer.zero_grad()
    
        outputs = model(images)
    
        loss = loss_function(outputs, labels)
        
        loss.backward()
    
        optimizer.step()
        
        if i % 2 == 0:
            print("Epoch: {} {}/{}".format(epoch,i,total_step))
            
# Train Accuracy
   
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in train_loader:
            
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            outputs = model(images)
            
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
            
            outputs = model(images)
            
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
    
    





































    

