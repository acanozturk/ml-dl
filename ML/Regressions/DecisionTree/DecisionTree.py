# Koltuk seviyesine göre fiyatlandırma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("original.csv", sep =";", header = None) # Dataseti okuduk ve değerlerin ; ile ayrıldığını belirttik

x = df.iloc[:,0].values.reshape(-1,1) # iloc, dataframe içniden 0. indexteki değerleri alır
y = df.iloc[:,1].values.reshape(-1,1)

# Decision tree regression
tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

a = np.arange(min(x), max(x), 0.01).reshape(-1,1) # Aralıklarda prediction yapabilmek için, tier aralıklarında fiyat sabit kalmalı

y_head = tree_reg.predict(a)

# Görselleştirme
plt.scatter(x, y, color = 'red')
plt.plot(a, y_head, color = 'green')
plt.xlabel("Seat Tier")
plt.ylabel("Price")
plt.show()