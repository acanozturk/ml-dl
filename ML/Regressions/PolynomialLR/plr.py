# Araba hızı-fiyat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("dataset.csv", sep =";") # Dataseti okuduk ve değerlerin ; ile ayrıldığını belirttik

x = df.fiyat.values.reshape(-1,1)
y = df.max_hız.values.reshape(-1,1)

lr = LinearRegression()
lr.fit(x ,y)

y_head = lr.predict(x)

# Polynomial feature oluşturma
plr = PolynomialFeatures(degree = 9) #2. dereceden polinom oluşturdu x_pol = plr.fit_transform(x) # xleri x2 ye çevirir, derece arttıkça doğruluk artar
x_pol = plr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_pol, y)

y_head2 = lr2.predict(x_pol)

plt.plot(x, y_head, color = "red", label = "LR")
plt.plot(x, y_head2, color = "black", label = "PLR")
plt.scatter(df.fiyat, df.max_hız)
plt.xlabel("Fiyat(Bin tl)")
plt.ylabel("Max. Hız")
plt.legend()
plt.show()


