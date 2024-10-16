import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Create a simple dataset
data = {
    'Size': [650, 785, 1200, 1500, 1800, 2100, 2500],  # Square feet
    'Price': [200000, 250000, 300000, 350000, 400000, 450000, 500000]  # Price in dollars
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)


print(df.head())  # Shows the first few rows of the data
print(df.describe())  # Gives you a summary of the data

#preprocess the data

# Features (input): Size of the house
X = df[['Size']]

# Target (output): Price of the house
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Create a linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print the predicted values and actual values for comparison
print("Predicted prices:", y_pred)
print("Actual prices:", y_test.values)

# Calculate the mean squared error (MSE) and R-squared (R^2) value
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R^2):", r_squared)

#To visualize the results
#plot the original data points
plt.scatter(X_test, y_test, color='blue', label='Actual prices')

#plot the regression line
plt.plot(X_test, y_pred, color='red', label='Predicted prices')

plt.title('House Prices vs Size')
plt.xlabel('House Size (Square Feet)')
plt.ylabel('House Price (Dollars)')
#plt.legend()