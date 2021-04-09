import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Datayı okur
data = pd.read_csv("cfdata.csv")

data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
data.tail()
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# Görselleştirme
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3) #alpha saydamlık verir
plt.scatter(B.radius_mean, B.texture_mean, color = "blue", label = "Benign", alpha = 0.3)
plt.xlabel("Radius mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)


# Normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

# Random forest classification
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print("Accuracy:",rf.score(x_test,y_test))

# Confusion Matrix
y_pred = rf.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true, y_pred)

# Heatmap yapacağız
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot = True, 
            linewidths = 0.8, 
            linecolor = "green", 
            fmt = ".0f", # float yazdırmak istemiyoruz
            ax = ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()





