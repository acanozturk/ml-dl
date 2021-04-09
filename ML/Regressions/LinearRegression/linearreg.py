# Deneyim-maaş

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Dataset.csv", sep =";") # Dataseti okuduk ve değerlerin ; ile ayrıldığını belirttik

# Dateset görselleştirme
plt.scatter(df.Deneyim, df.Maaş)
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()

# Lineer regression, y = a + b.x, a = bias b = coefficient
lr = LinearRegression()

x = df.Deneyim.values.reshape(-1,1) # Normalde x shape(15,) olarak görünür fakat sklearn işlemleri için (15,1) yapmamaız gerekir
y = df.Maaş.values.reshape(-1,1)

print("x shape:", x.shape)
print("y shape:", y.shape)

lr.fit(x, y) # Fit çizgisi oluşturuldu

# a(bias) bulma yöntemleri
a = lr.predict([[0]])
print("a:", a)

a_ = lr.intercept_
print("a:", a_)

# b(coef.) bulma
b = lr.coef_
print("b:", b)

# Maaş = 1437 + (503 x Deneyim)i maaşları predict edebiliriz
maas_predict = lr.predict([[12]])
print("Maaş:", maas_predict)

# Fit line çizdirme
array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
print("array shape:", array.shape)

y_head = lr.predict(array) # y_headleri tahmin ederek Fit line oluşturur
plt.plot(array, y_head, color = "red") 
plt.scatter(x,y) # Gerçek data
plt.show()


























































