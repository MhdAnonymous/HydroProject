import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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

# Define models
models = {
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_2000_strat, y_train_2000_strat)
    y_pred = model.predict(X_test_2000_strat)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test_2000_strat, y_pred),
        'Precision': precision_score(y_test_2000_strat, y_pred),
        'Recall': recall_score(y_test_2000_strat, y_pred),
        'F1 Score': f1_score(y_test_2000_strat, y_pred),
        'Confusion Matrix': confusion_matrix(y_test_2000_strat, y_pred)
    }

results
