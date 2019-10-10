from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import numpy as np 

raw_pulsar_data = pd.read_csv("pulsar_stars.csv")

print("--- SAMPLE DATA ---")
print(raw_pulsar_data.head(), "\n")

print("--- DATA INFO ---")
print(raw_pulsar_data.describe(), "\n")

print("--- FEATURES ---")
print(raw_pulsar_data.columns)