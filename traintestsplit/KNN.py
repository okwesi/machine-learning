import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#READING THE DATA
data = pd.read_csv('car.data')
    #print(data)
X = data[[
        'buying',
          'maintenance',
          'safety'
          ]].values

y = data[['class']]

#CONVERTING THE DATA
le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i ] = le.fit_transform(X[:, i])

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping) # Mapping
y = np.array(y)



knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.20)

knn.fit(X_train, y_train)

y_prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)

#print(f" y_prediction : {y_prediction}")
print(f"Accuracy : {accuracy}")

print(f"actual value {y[234]}")
print(f"predicted value : {knn.predict(X)[20]}")
