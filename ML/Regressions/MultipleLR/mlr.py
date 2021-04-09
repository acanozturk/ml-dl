# Deneyim-yaş-maaş


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("mlr.csv", sep =";") # Dataseti okuduk ve değerlerin ; ile ayrıldığını belirttik

'''
Sıralı nesnelerdeki dilimleme işlemleri Pandas serilerinde de kullanılabilir. 
Dilimleme için iki seçenek var: indeksin sırası ve kendisi.
iloc indeksin sırasıyla, loc indeksin kendisiyle işlem yapmaya olanak verir. 
loc metotunda ilk eleman da son eleman da dilimlemeye dahil edilir.
'''
x = df.iloc[:,[0,2]].values
y = df.Maaş.values.reshape(-1,1)

print("x shape:", x.shape)
print("y shape:", y.shape)

mlr = LinearRegression()
mlr.fit(x, y)

a = mlr.intercept_
print("a:", a)

b1_2 = mlr.coef_
print("b1, b2:", b1_2)


# Maaş = -1164 + (335 x Deneyim ) + (121 x Yaş)
for d in range(0, 16, 1):
    for y in range(22, 43, 1):
        array = np.array([[d,y]])
        maas_predict = mlr.predict(array)
        print(maas_predict)
    
