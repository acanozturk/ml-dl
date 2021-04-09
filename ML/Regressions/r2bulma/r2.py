# Tribün-Koltuk Değeri

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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

y_head = rfr.predict(x)

r2score = r2_score(y, y_head)
print("r2 score:", r2score)
 