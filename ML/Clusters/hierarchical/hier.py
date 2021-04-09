import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram # Dendogram çizdirmek için
from sklearn.cluster import AgglomerativeClustering

# Dataset oluşturma 3 farklı class olacak
x1 = np.random.normal(25, 5, 100) # 25 ortalama 5 sigmaya(25+5,25-5) sahip 1000 değer üret
y1 = np.random.normal(25, 5, 100)

x2 = np.random.normal(55, 5, 100) 
y2 = np.random.normal(60, 5, 100)

x3 = np.random.normal(55, 5, 100) 
y3 = np.random.normal(15, 5, 100)

# X ve Y leri birleştirdik
x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)

# X ve Y yi birleştirdik
dictionary = {"x":x, "y":y}

# Dataframe oluşturma
data = pd.DataFrame(dictionary)
data.describe()

# Dendrogram
merg = linkage(data, method = "ward")
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
clusters = hc.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "blue")
plt.show()