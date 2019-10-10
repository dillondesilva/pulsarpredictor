# Author: Dillon de Silva

# Description: Program which creates a
# machine learning model of whether a given
# neutron star is also of subclass "Pulsar"
# based off its radio emissions

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 
import numpy as np 

raw_pulsar_data = pd.read_csv("pulsar_stars.csv")

# Data Visualization
print("--- SAMPLE DATA ---")
print(raw_pulsar_data.head(), "\n")

print("--- DATA INFO ---")
print(raw_pulsar_data.describe(), "\n")

print("--- FEATURES ---")
print(raw_pulsar_data.columns, "\n")

features = [" Mean of the integrated profile",
      " Standard deviation of the integrated profile",
      " Excess kurtosis of the integrated profile",
      " Skewness of the integrated profile", 
      " Mean of the DM-SNR curve",
      " Standard deviation of the DM-SNR curve",
      " Excess kurtosis of the DM-SNR curve", 
      " Skewness of the DM-SNR curve"]

# Splitting data set into training
# and test data
X = raw_pulsar_data[features]
y = raw_pulsar_data["target_class"]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

# Creating a Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

# Model Validation
test_pred = rf_model.predict(test_X)

mae = mean_absolute_error(test_pred, test_y)
print("Mean Absolute Error:", mae)

# Outputting first couple of rows
print(test_pred[:5])
print(test_y[:5])