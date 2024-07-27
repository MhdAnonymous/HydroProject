import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
 
# Load the PS2 and FS1 data
ps2_data_path = os.path.join('data_subset', 'PS2.txt')
fs1_data_path = os.path.join('data_subset', 'FS1.txt')
ps2_data = pd.read_csv(ps2_data_path, delimiter='\t', header=None)
fs1_data = pd.read_csv(fs1_data_path, delimiter='\t', header=None)

# Load the profile data
profile_data_path = os.path.join('data_subset', 'profile.txt')
profile_data = pd.read_csv(profile_data_path, delimiter='\t', header=None)
profile_data.columns = [
    'Cooler Condition (%)',
    'Valve Condition (%)',
    'Internal Pump Leakage',
    'Hydraulic Accumulator Pressure (bar)',
    'Stable Flag'
]

# Define the target
y = profile_data['Valve Condition (%)']

# Combine PS2 and FS1 data for input features
ps2_features = ps2_data.mean(axis=1).to_frame(name='PS2_Mean')
fs1_features = fs1_data.mean(axis=1).to_frame(name='FS1_Mean')

# Combine features
X = pd.concat([ps2_features, fs1_features], axis=1)

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

# Evaluate each model and save them
results = {}
for name, model in models.items():
    model.fit(X_train_2000_strat, y_train_2000_strat)
    y_pred = model.predict(X_test_2000_strat)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test_2000_strat, y_pred),
        'Precision': precision_score(y_test_2000_strat, y_pred, average='weighted'),
        'Recall': recall_score(y_test_2000_strat, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test_2000_strat, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test_2000_strat, y_pred)
    }
    
    # Save the model
    model_filename = f'{name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model {name} saved as {model_filename}")

print(results)
# Load the PS2 and FS1 data
ps2_data_path = os.path.join('data_subset', 'PS2.txt')
fs1_data_path = os.path.join('data_subset', 'FS1.txt')
ps2_data = pd.read_csv(ps2_data_path, delimiter='\t', header=None)
fs1_data = pd.read_csv(fs1_data_path, delimiter='\t', header=None)

# Load the profile data
profile_data_path = os.path.join('data_subset', 'profile.txt')
profile_data = pd.read_csv(profile_data_path, delimiter='\t', header=None)
profile_data.columns = [
    'Cooler Condition (%)',
    'Valve Condition (%)',
    'Internal Pump Leakage',
    'Hydraulic Accumulator Pressure (bar)',
    'Stable Flag'
]

# Define the target
y = profile_data['Valve Condition (%)']

# Combine PS2 and FS1 data for input features
ps2_features = ps2_data.mean(axis=1).to_frame(name='PS2_Mean')
fs1_features = fs1_data.mean(axis=1).to_frame(name='FS1_Mean')

# Combine features
X = pd.concat([ps2_features, fs1_features], axis=1)

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

# Convert X_test_2000_strat to a DataFrame for better readability
X_test_2000_strat_df = pd.DataFrame(X_test_2000_strat)

# Print the first few rows of the test data
print("First few rows of X_test_2000_strat:")
print(X_test_2000_strat_df.head())

# Function to load a model and predict on specific test data
def load_model_and_predict(model_filename, X_test):
    # Load the model
    model = joblib.load(model_filename)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    return y_pred

# Example usage
model_filename = 'svm_model.pkl'  # Change this to the desired model file
y_pred = load_model_and_predict(model_filename, X_test_2000_strat)

# Print the predictions
print(f"Predictions using {model_filename}:")
print(y_pred)
