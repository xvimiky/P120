from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import csv
import pandas as pd
from sklearn import datasets


wine = datasets.load_wine()


print("Features: ", wine.feature_names)

print("Labels: ", wine.target_names)

x = wine.data
y = wine.target 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 109)

gb = GaussianNB()
gb.fit(x_train , y_train)


y_pred = gb.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ",accuracy)


