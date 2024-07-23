# Re-import necessary libraries and redefine variables to ensure a fresh start
import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
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

# Create aggregated statistical features (mean, median, std) for each cycle
X['Mean Cooler Condition'] = np.mean(profile_data['Cooler Condition (%)'])
X['Median Cooler Condition'] = np.median(profile_data['Cooler Condition (%)'])
X['Std Cooler Condition'] = np.std(profile_data['Cooler Condition (%)'])

X['Mean Pump Leakage'] = np.mean(profile_data['Internal Pump Leakage'])
X['Median Pump Leakage'] = np.median(profile_data['Internal Pump Leakage'])
X['Std Pump Leakage'] = np.std(profile_data['Internal Pump Leakage'])

X['Mean Accumulator Pressure'] = np.mean(profile_data['Hydraulic Accumulator Pressure (bar)'])
X['Median Accumulator Pressure'] = np.median(profile_data['Hydraulic Accumulator Pressure (bar)'])
X['Std Accumulator Pressure'] = np.std(profile_data['Hydraulic Accumulator Pressure (bar)'])

X['Stable Flag Count'] = profile_data['Stable Flag'].sum()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create polynomial and interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data into training and testing sets with 2000 samples for training and the rest for testing
X_train_poly, X_test_poly, y_train_poly, y_test_poly = X_poly[:2000], X_poly[2000:], y[:2000], y[2000:]

# Train and evaluate Random Forest Classifier with the specified training and testing split
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_poly, y_train_poly)
y_pred_rf = rf_model.predict(X_test_poly)

accuracy_rf = accuracy_score(y_test_poly, y_pred_rf)
precision_rf = precision_score(y_test_poly, y_pred_rf)
recall_rf = recall_score(y_test_poly, y_pred_rf)
f1_rf = f1_score(y_test_poly, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test_poly, y_pred_rf)

# Display evaluation results for the Random Forest model
rf_evaluation_results_custom_split = {
    'Accuracy': accuracy_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'F1 Score': f1_rf,
    'Confusion Matrix': conf_matrix_rf
}

rf_evaluation_results_custom_split
