# Author: Dillon de Silva

# Description: Program which creates a
# machine learning model of whether a given
# neutron star is also of subclass "Pulsar"
# based off its radio emissions

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd 
import numpy as np 
import sys

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

def build_all_models():
  build_random_forest_regressor_model()
  build_k_nearest_neighbours_model()

def build_specified_model(model):
  if model == "rf":
    build_random_forest_regressor_model()
  elif model == "knn":
    build_k_nearest_neighbours_model()
  elif model == "nn":
    build_neural_network_model()


def build_neural_network_model():
  print("\n--- CREATING NEURAL NETWORK MODEL ---") 
  # Creating a Neural Network MLP Regressor Model
  mlp_model = MLPRegressor(random_state=1)
  mlp_model.fit(train_X, train_y)

  # Model Validation
  test_pred = np.round(abs(mlp_model.predict(test_X)))

  mae = mean_absolute_error(test_pred, test_y)
  r2 = r2_score(test_pred, test_y)

  print("\n" + "Mean Absolute Error:", mae)
  print("R2 Score:", r2) 

  # Outputting first couple of rows
  print(test_pred[:5])
  print(test_y[:5])

  # Model Improvement
  # Creating discrete hyperparameter amounts to trial
  print("\n--- BEGINNING MODEL IMPROVEMENTS ---")
  solvers = ["adam", "lbfgs", "sgd"]

  print("\n--- ADJUSTING SOLVER ---")

  best_solver_data = ["", 0]
  for solver in solvers:
    mlp_model = MLPRegressor(solver=solver, random_state=1)
    mlp_model.fit(train_X, train_y)
    preds = np.round(abs(mlp_model.predict(test_X)))
    score = r2_score(preds, test_y)
    print("\nSolver:", solver)
    print("R2 Score:", score)
    if score > best_solver_data[1]:
      best_solver_data = [solver, score]

  best_solver = best_solver_data[0]
  print("\nOptimal solver:", best_solver)

def build_k_nearest_neighbours_model():
  print("\n--- CREATING K NEAREST NEIGHBOURS REGRESSOR MODEL ---") 
  # Creating a K Nearest Neighbours Regressor Model
  knn_model = KNeighborsRegressor()
  knn_model.fit(train_X, train_y)

  # Model Validation
  test_pred = knn_model.predict(test_X)

  mae = mean_absolute_error(test_pred, test_y)
  r2 = r2_score(test_pred, test_y)

  print("\n" + "Mean Absolute Error:", mae)
  print("R2 Score:", r2) 

  # Outputting first couple of rows
  print(test_pred[:5])
  print(test_y[:5])

  # Model Improvement
  # Creating discrete hyperparameter amounts to trial
  print("\n--- BEGINNING MODEL IMPROVEMENTS ---")
  n_neighbors = [1, 2, 3, 5, 10, 15, 50, 100, 1000]
  leaf_sizes = [2, 5, 10, 30, 50, 100]
  algorithms = ["auto", "ball_tree", "kd_tree", "brute"]

  print("\n--- ADJUSTING N NEIGHBOURS ---")

  best_n_neighbors_data = [0, 0]
  for n_neighbor in n_neighbors:
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbor)
    knn_model.fit(train_X, train_y)
    preds = knn_model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nN Neighbors:", n_neighbor)
    print("R2 Score:", score)
    if score > best_n_neighbors_data[1]:
      best_n_neighbors_data = [n_neighbor, score]

  best_n_neighbors = best_n_neighbors_data[0]
  print ("\nOptimal amount of n neighbours:", best_n_neighbors)

  print("\n--- ADJUSTING LEAF SIZE ---")

  best_leaf_size_data = [0, 0]
  for leaf_size in leaf_sizes:
    knn_model = KNeighborsRegressor(leaf_size=leaf_size)
    knn_model.fit(train_X, train_y)
    preds = knn_model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nLeaf size:", leaf_size)
    print("R2 Score:", score)
    if score > best_leaf_size_data[1]:
      best_leaf_size_data = [leaf_size, score]

  best_leaf_size = best_leaf_size_data[0]
  print ("\nOptimal leaf size:", best_leaf_size)

  print("\n--- FINDING OPTIMAL ALGORITHM ---")

  best_algorithm_data = ["", 0]
  for algorithm in algorithms:
    knn_model = KNeighborsRegressor(algorithm=algorithm)
    knn_model.fit(train_X, train_y)
    preds = knn_model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nAlgorithm:", algorithm)
    print("R2 Score:", score)
    if score > best_algorithm_data[1]:
      best_algorithm_data = [algorithm, score]

  best_algorithm = best_algorithm_data[0]
  print ("\nOptimal algorithm:", best_algorithm)

  print("---CREATING FINAL MODEL ---")
  model = KNeighborsRegressor(n_neighbors=best_n_neighbors, leaf_size=best_leaf_size, algorithm=best_algorithm)
  model.fit(train_X, train_y)
  preds = model.predict(test_X)
  score = r2_score(preds, test_y)
  print("R2 Score:", score)

def build_random_forest_regressor_model():
  print("\n--- CREATING RANDOM FOREST REGRESSOR MODEL ---") 
  # Creating a Random Forest Regressor Model
  rf_model = RandomForestRegressor(random_state=1)
  rf_model.fit(train_X, train_y)

  # Model Validation
  test_pred = rf_model.predict(test_X)

  mae = mean_absolute_error(test_pred, test_y)
  r2 = r2_score(test_pred, test_y)

  print("\n", "Mean Absolute Error:", mae)
  print("\n", "R2 Score:", r2)

  # Outputting first couple of rows
  print(test_pred[:5])
  print(test_y[:5])

  # Model Improvement
  # Creating discrete hyperparameter amounts to trial
  print("\n--- BEGINNING MODEL IMPROVEMENTS ---")

  max_leaf_nodes = [10, 100, 1000, 10000]
  min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  n_estimators = [10, 100, 1000]
  max_features = ["auto", "sqrt", "log2"]

  print("\n--- ADJUSTING MAX LEAF NODES ---")

  best_max_leaf_nodes_data = [0, 0]
  for max_leaf_node in max_leaf_nodes:
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_node, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    score = r2_score(preds, test_y) 
    print("\nMax Leaf Nodes:", max_leaf_node)
    print("R2 Score:", score)
    if score > best_max_leaf_nodes_data[1]:
      best_max_leaf_nodes_data = [max_leaf_node, score]

  best_max_leaf_nodes = best_max_leaf_nodes_data[0]
  print ("\nOptimal amount of max leaf nodes:", best_max_leaf_nodes)

  print("\n--- ADJUSTING MIN SAMPLES PER LEAF ---")

  # best_min_samples_leaf is an array with two values
  # 1) Optimal amount of Minimum Samples a leaf
  # 2) R2 score of the optimal amount
  best_min_samples_leaf_data = [0, 0]
  for min_samples_leaf in min_samples_leaf:
    model = RandomForestRegressor(min_samples_leaf=min_samples_leaf, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nMin Samples Per Leaf:", min_samples_leaf)
    print("R2 Score:", score)
    if score > best_min_samples_leaf_data[1]:
      best_min_samples_leaf_data = [min_samples_leaf, score]

  best_min_samples_leaf = best_min_samples_leaf_data[0] 
  print("\nOptimal amount of minimum samples a leaf:", best_min_samples_leaf)

  print("--- ADJUSTING MAX FEATURES ---")

  best_max_feature_data = ["", 0]
  for max_feature in max_features:
    model = RandomForestRegressor(max_features=max_feature, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nMax Feature:", max_feature)
    print("R2 Score:", score)
    if score > best_max_feature_data[1]:
      best_max_feature_data = [max_feature, score]

  best_max_feature = best_max_feature_data[0] 
  print("\nOptimal Max Feature:", best_max_feature)

  print("--- ADJUSTING N ESTIMATORS ---")

  best_n_est_data = [0, 0]
  for n_estimator in n_estimators:
    model = RandomForestRegressor(n_estimators=n_estimator, random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    score = r2_score(preds, test_y)
    print("\nN Estimator:", n_estimator)
    print("R2 Score:", score)
    if score > best_min_samples_leaf_data[1]:
      best_n_est_data = [n_estimator, score]

  best_n_estimator = best_n_est_data[0] 
  print("\nOptimal N Estimator:", best_n_estimator)

  print("---CREATING FINAL MODEL ---")
  model = RandomForestRegressor(max_leaf_nodes=best_max_leaf_nodes, n_estimators=best_n_estimator, random_state=1)
  model.fit(train_X, train_y)
  preds = model.predict(test_X)
  score = r2_score(preds, test_y)
  print("\nMin Samples Per Leaf:", min_samples_leaf)
  print("R2 Score:", score)

NO_MODEL_ERR_MSG = "No model build type specified. Model build type must be specified when running program in the syntax, '--model=$(MODEL_TYPE)'."
INVALID_MODEL_ERR_MSG = "Invalid model build type specified. Valid model build types include:\n['rf', 'knn']"
# Obtaining model type build specification
# from passed parameter

allowed_models = ["rf", "knn", "nn"]
num_of_args = len(sys.argv)

if num_of_args == 1:
  print("No model type specified. Building all models...")
  build_all_models()
else:
  arg, model = sys.argv[1].split("=")
  if arg != "-model":
    raise Exception(NO_MODEL_ERR_MSG)
  elif model not in allowed_models:
    raise Exception(INVALID_MODEL_ERR_MSG)

  build_specified_model(model)