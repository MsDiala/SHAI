# Diamond Price Prediction

This program utilizes machine learning techniques to predict the price of diamonds based on various features. It implements Linear Regression and Random Forest models to perform the prediction. The dataset used in this program contains information about the carat, clarity, color, dimensions, and other attributes of the diamonds.

## Usage
1. Ensure that the required libraries (scikit-learn, pandas, numpy) are installed.
2. Prepare the dataset in CSV format with the necessary features and target variable.
3. Configure the data file path and other parameters in the code.
4. Run the program to train the models and evaluate their performance.

## Author
Your Name

## Date
Date

## Installation
Make sure the following libraries are installed:
- scikit-learn
- pandas
- numpy

## Dataset
The program expects the dataset to be in CSV format with the following columns:
- carat: Weight of the diamond
- clarity: Degree of diamond clarity
- color: Diamond color
- x: Length in mm
- y: Width in mm
- z: Depth in mm
- depth: Total depth percentage
- table: Width of top of diamond relative to widest point
- cut: Quality of the diamond cut
- price: Price of the diamond (target variable)

## Usage Example
# Import necessary libraries

``
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
``


# Load the dataset
``
data = pd.read_csv("diamonds.csv")
``
# Data preprocessing
# TODO: Perform any necessary data cleaning, feature engineering, or encoding

# Split the data into training and testing sets
``
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
``

# Train the Linear Regression model
``
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
``

# Make predictions using Linear Regression model
``
lr_pred = lr_model.predict(X_test)
``

# Calculate RMSE for Linear Regression
``
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print("Linear Regression RMSE:", lr_rmse)
``

# Train the Random Forest model
``
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
``

# Make predictions using Random Forest model
``
rf_pred = rf_model.predict(X_test)
``

# Calculate RMSE for Random Forest
``
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print("Random Forest RMSE:", rf_rmse)
``
