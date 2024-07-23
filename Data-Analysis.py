import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Re-extract the contents of the zip file since the environment was reset
zip_path = '/mnt/data/data_subset.zip'
extracted_path = '/mnt/data/data_subset'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Load the profile data
profile_data_path = os.path.join(extracted_path, 'data_subset', 'profile.txt')
profile_data = pd.read_csv(profile_data_path, delimiter='\t', header=None)
profile_data.columns = [
    'Cooler Condition (%)',
    'Valve Condition (%)',
    'Internal Pump Leakage',
    'Hydraulic Accumulator Pressure (bar)',
    'Stable Flag'
]
profile_data['Valve Optimal'] = profile_data['Valve Condition (%)'] == 100

# Features and target
X = profile_data.drop(['Valve Condition (%)', 'Valve Optimal'], axis=1)
y = profile_data['Valve Optimal']

# Feature Engineering
# Aggregated statistical features (mean, median, std) for each cycle
X['Mean Cooler Condition'] = profile_data['Cooler Condition (%)'].mean()
X['Median Cooler Condition'] = profile_data['Cooler Condition (%)'].median()
X['Std Cooler Condition'] = profile_data['Cooler Condition (%)'].std()

X['Mean Pump Leakage'] = profile_data['Internal Pump Leakage'].mean()
X['Median Pump Leakage'] = profile_data['Internal Pump Leakage'].median()
X['Std Pump Leakage'] = profile_data['Internal Pump Leakage'].std()

X['Mean Accumulator Pressure'] = profile_data['Hydraulic Accumulator Pressure (bar)'].mean()
X['Median Accumulator Pressure'] = profile_data['Hydraulic Accumulator Pressure (bar)'].median()
X['Std Accumulator Pressure'] = profile_data['Hydraulic Accumulator Pressure (bar)'].std()

X['Stable Flag Count'] = profile_data['Stable Flag'].sum()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create polynomial and interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Perform a stratified split with 2000 samples for training and the rest for testing
strat_split_2000 = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=42)
for train_index, test_index in strat_split_2000.split(X_poly, y):
    X_train_2000_strat, X_test_2000_strat = X_poly[train_index], X_poly[test_index]
    y_train_2000_strat, y_test_2000_strat = y[train_index], y[test_index]

# Define the parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Perform GridSearchCV for SVM
grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_2000_strat, y_train_2000_strat)

# Get the best estimator
best_svm = grid_svm.best_estimator_

# Evaluate the best SVM model on the test set
y_pred_best_svm = best_svm.predict(X_test_2000_strat)

accuracy_best_svm = accuracy_score(y_test_2000_strat, y_pred_best_svm)
precision_best_svm = precision_score(y_test_2000_strat, y_pred_best_svm)
recall_best_svm = recall_score(y_test_2000_strat, y_pred_best_svm)
f1_best_svm = f1_score(y_test_2000_strat, y_pred_best_svm)
conf_matrix_best_svm = confusion_matrix(y_test_2000_strat, y_pred_best_svm)

# Display evaluation results for the best SVM model after hyperparameter tuning
best_svm_evaluation_results = {
    'Best Parameters': grid_svm.best_params_,
    'Accuracy': accuracy_best_svm,
    'Precision': precision_best_svm,
    'Recall': recall_best_svm,
    'F1 Score': f1_best_svm,
    'Confusion Matrix': conf_matrix_best_svm
}

print(best_svm_evaluation_results)
