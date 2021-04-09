import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Datayı okur
data = pd.read_csv("svmdata.csv")


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

# SVM
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)

print("Accuracy:",svm.score(x_test,y_test))