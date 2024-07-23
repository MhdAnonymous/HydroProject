from sklearn.feature_selection import RFE

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

# Feature Selection using RFE with Random Forest
rfe_selector = RFE(RandomForestClassifier(random_state=42), n_features_to_select=10, step=1)
X_selected = rfe_selector.fit_transform(X_poly, y)

# Perform a stratified split with 2000 samples for training and the rest for testing
strat_split_2000 = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=42)
for train_index, test_index in strat_split_2000.split(X_selected, y):
    X_train_2000_strat, X_test_2000_strat = X_selected[train_index], X_selected[test_index]
    y_train_2000_strat, y_test_2000_strat = y[train_index], y[test_index]

# Train and evaluate Random Forest Classifier with the stratified 2000 training samples
rf_model_2000_strat = RandomForestClassifier(random_state=42)
rf_model_2000_strat.fit(X_train_2000_strat, y_train_2000_strat)
y_pred_rf_2000_strat = rf_model_2000_strat.predict(X_test_2000_strat)

accuracy_rf_2000_strat = accuracy_score(y_test_2000_strat, y_pred_rf_2000_strat)
precision_rf_2000_strat = precision_score(y_test_2000_strat, y_pred_rf_2000_strat)
recall_rf_2000_strat = recall_score(y_test_2000_strat, y_pred_rf_2000_strat)
f1_rf_2000_strat = f1_score(y_test_2000_strat, y_pred_rf_2000_strat)
conf_matrix_rf_2000_strat = confusion_matrix(y_test_2000_strat, y_pred_rf_2000_strat)

# Display evaluation results for the enhanced Random Forest model
enhanced_rf_evaluation_results_2000_strat = {
    'Accuracy': accuracy_rf_2000_strat,
    'Precision': precision_rf_2000_strat,
    'Recall': recall_rf_2000_strat,
    'F1 Score': f1_rf_2000_strat,
    'Confusion Matrix': conf_matrix_rf_2000_strat
}
enhanced_rf_evaluation_results_2000_strat