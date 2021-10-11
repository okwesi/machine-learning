from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt


boston_data = datasets.load_wine()


#featurs / labels
X = boston_data.data
y = boston_data.target

num = X.shape


l_reg = linear_model.LinearRegression()


plt.scatter(X.T[num[1]-1], y)


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2)

model = l_reg.fit(X_train, y_train)

predictions = model.predict(X_test)


for test in X_test:
    i = 1
    print(f"Test {i} : {test}")
    i = i + 1
for prediction in predictions:
    i = 1
    print(f"Prediction {i} :  {prediction}")
    i = i + 1

print(f"Accuracy : {l_reg.score(X, y)}")
print(f"coedd : {l_reg.coef_}")
