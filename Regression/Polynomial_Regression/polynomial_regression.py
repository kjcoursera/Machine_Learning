# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)

# Fitting Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lm_poly = LinearRegression()
lm_poly.fit(X_poly,y)


#Visualizing Linear Regression results

plt.scatter(X,y,color = 'red')
plt.plot(X,lm.predict(X),color='blue')
plt.title('True or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualizing Polynomial Regression results

plt.scatter(X,y, color= 'green')
plt.plot(X,lm_poly.predict(X_poly),color= 'blue')
plt.title('True or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color= 'green')
plt.plot(X_grid,lm_poly.predict(poly_reg.fit_transform(X_grid)),color= 'blue')
plt.title('True or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#y_pred = lm.predict(6.5)
#print("Linear Regression Results:", y_pred)
#
#y_pred_poly = lm_poly.predict(poly_reg.fit_transform(6.5))
#print("Linear Regression Results:", y_pred_poly)
