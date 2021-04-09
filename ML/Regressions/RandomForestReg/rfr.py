# Tribün-Koltuk Değeri

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("forest.csv", sep =";", header = None)

x = df.iloc[:,0].values.reshape(-1,1) # iloc, dataframe içniden 0. indexteki değerleri alır
y = df.iloc[:,1].values.reshape(-1,1)

# Random Forest RegressioN
'''
n_estimators: Forest içinde kaç tane tree olacak
random_state: Kodu her fit edişte bu sayıya göre böl, aynı random değerleri ver
'''
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42) 

rfr.fit(x,y)

# Görselleştirme

a = np.arange(min(x), max(x), 0.01).reshape(-1,1)

y_head = rfr.predict(a)

plt.scatter(x,y, color = "red")
plt.plot(a, y_head, color = "green")
plt.xlabel("Seat Tier")
plt.ylabel("Price")
plt.show()
