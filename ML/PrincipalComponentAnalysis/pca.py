from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA

#####################################################
# DATAFRAME OLUŞTUMA
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = feature_names)
df["classes"] = y # Labellar

x = data # Zaten nparray olduğu için values() demeye gerek yok
#####################################################


#####################################################
# PCA #
pca = PCA(n_components = 2, # Datayı kaç boyuta düşüreceğimizi burada belirtiriz
          whiten = True) # Datayı normalize eder

x_pca = pca.fit_transform(x) # Boyut dönüştürme işlemi burada yapılır
#####################################################

df["p1"] = x_pca[:,0]  # Principle Comp.
df["p2"] = x_pca[:,1]  # Secondary Comp.

color = ["red", "green", "blue"]

for each in range(3): # 3 farklı çiçek türümüz var
    plt.scatter(df.p1[df.classes == each ], 
                df.p2[df.classes == each ], 
                color = color[each], 
                label  = iris.target_names[each])
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()





