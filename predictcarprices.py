import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Engine Size': [1500, 1800, 2000, 2200, 2500, 3000, 3500, 4000],  # Engine size in cc
    'Price': [15000, 18000, 20000, 22000, 25000, 30000, 35000, 40000]  # Price in dollars
}

df= pd.DataFrame(data)

print (df.head())
print (df.describe())

X = df[['Engine Size']]
y = df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted prices:", y_pred)
print("Actual Prices:",y_test.values)

mse =  mean_squared_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Price ($)')
plt.title('Car Prices vs. Engine Size')
plt.legend()
plt.show()

