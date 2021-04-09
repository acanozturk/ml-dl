# İyi-kötü huylu tümör

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("logresdata.csv")
#print(data.info()) # Data içindeki featureları görmek için

'''
data.drop: İstenmeyen featureları datadan kaldırmaya yarar
axis=1: Sütunu silmeye yarar, bunu yazmazsak satırı siler
inplace=True: Datayı sildikten sonra tekrar dataseti günceller
'''
data.drop(["Unnamed: 32","id"], axis = 1, inplace = True)

# Diagnosis feature'ı içindeki dataları B ise 0 ve M ise 1 yapmak istiyoruz
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)

# Normalization. tüm featureları 0-1 arasına çeker
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values
 
# Train-Test split
'''
Kolaylık olsun diye featurların ve sampleların yerini değiştirdik. 
Toplamda 30 feature ve 569 sample var. 
Samplerın 0.8i train ve 0.2si test için ayrıldı.
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Satır-Sütün yerini değiştirir.
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



# PARAMETER INITIALIZE
def initialize_weigths_and_bias(dimension):
    w = np.full((dimension, 1), 0.01) # Dimensiona(Bizim dimension = 30) bağlı, başlangıç değerleri 0.01 olan ones matrisi yapar
    b = 0.0
    return w,b



# SIGMOID FUNCTION
def sigmoid(z):    
    y_head = 1/(1 + np.exp(-z))
    return y_head



# FORWARD-BACKWARD PROPAGATION
def forward_backward_propagation(w, b, x_train, y_train):
    # Forward Prog.
    z = np.dot(w.T, x_train) + b # Weightleri featurelarla çarpmamız lazım. Bu matris çarpımını yapabilmek için transpoze alınır.
    y_head = sigmoid(z) # z değerini sigmoide sokarak 0-1 arasına çekeriz ve y_head elde ederiz
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head) # Loss fonksiyonu böyle hesaplanıyor
    cost = np.sum(loss) / x_train.shape[1] # Lossların toplamı costu verir. Bölerek normalize ettik
    
    # Backward Prog.
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias} # Weight ve Bias türevlerini depolar
    
    return cost,gradients



# PARAMETER UPDATE
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = [] # Tüm costları depolamak için array
    cost_list2 = [] # Adım sayısında gösterilecek costlar için array
    index = []
    
    # Learning parametrelerini iterasyon sayısı kez günceller
    for i in range(number_of_iteration):
        # Forward-Backward Propagation yaparak cost ve gradient bulunur
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        # Parametreleri güncelleme
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 50 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
            
    
    # Güncellenmiş parametreler
    parameters = {"weight":w,"bias":b}
    
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation ="vertical")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters,gradients,cost_list
    


# PREDICTION
def predict(w, b, x_test):
    # Test datasına forward progagation uygulanır ve 0-1 arasına çekilir
    z = sigmoid(np.dot(w.T, x_test) + b)
    
    y_prediction = np.zeros((1, x_test.shape[1]))        
    
    # Eğer prediction 0.5'ten büyükse 1 küçükse 0
    for i in range(z.shape[1]):
        if z[0:i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    return y_prediction



# LOGISTIC REGRESSION
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):   
    
    # Initialize
    dimension = x_train.shape[0] 
    w,b = initialize_weigths_and_bias(dimension)
    
    # Update
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    # Predict
    y_prediction_test = predict(parameters["weight"],  parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    
    # Doğruluk oranları
    print("Train Acc: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Train Acc: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
  
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 500)
