import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset oluşturma 3 farklı class olacak
x1 = np.random.normal(25, 5, 1000) # 25 ortalama 5 sigmaya(25+5,25-5) sahip 1000 değer üret
y1 = np.random.normal(25, 5, 1000)

x2 = np.random.normal(55, 5, 1000) 
y2 = np.random.normal(60, 5, 1000)

x3 = np.random.normal(55, 5, 1000) 
y3 = np.random.normal(15, 5, 1000)

# X ve Y leri birleştirdik
x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

# X ve Y yi birleştirdik
dictionary = {"x":x, "y":y}

# Dataframe oluşturma
data = pd.DataFrame(dictionary)
data.describe()

# Labellarımız olamdğığ için kmeans alg. datayı böyle görür
plt.scatter(x1,y1, color = "black")
plt.scatter(x2,y2, color = "black")
plt.scatter(x3,y3, color = "black")
plt.show()

# kmeans clustering
wcss = []

for k in range(1,20):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,20), wcss)  
plt.xlabel("Number of k ")
plt.ylabel("WCSS")
plt.show() 

# En iyi k 3 oalrak gördük. k=3 içn model
kmeans2 = KMeans(n_clusters = 3)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "blue")
plt.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], color = "yellow")

