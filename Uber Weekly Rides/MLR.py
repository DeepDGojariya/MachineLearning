import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

print("Train Score : ",regressor.score(X_train,y_train))
print("Test Score : ",regressor.score(X_test,y_test))

pickle.dump(regressor,open('taxi.pkl','wb'))

model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]]))