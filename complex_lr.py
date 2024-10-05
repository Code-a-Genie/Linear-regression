import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'Engine Size': [1500, 1800, 2000, 2200, 2500, 3000, 3500, 4000],
    'Mileage': [30, 28, 25, 20, 18, 15, 12, 10],
    'Age': [3, 2, 4, 5, 3, 6, 7, 8],
    'Horsepower': [130, 150, 170, 190, 200, 240, 270, 300],
    'Price': [15000, 18000, 20000, 22000, 25000, 30000, 35000, 40000]
}

df = pd.DataFrame(data)

X = df[['Engine Size', 'Mileage', 'Age', 'Horsepower' ]]
y = df['Price']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)    

X_train