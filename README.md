**This project** involves the analysis of hydraulic system data obtained from a hydraulic test rig. The test rig consists of a primary working circuit and a secondary cooling-filtration circuit connected via an oil tank. The system repeatedly performs constant load cycles, measuring various process values such as pressures, volume flows, and temperatures, while the condition of four hydraulic components (cooler, valve, pump, and accumulator) is varied.

**Project Structure**

The project is divided into several sections, each implemented in its own Python script:

**data_analysis.py**: Main script for data analysis.

**RandomForest.py**: Script for initial data processing and training of a Random Forest model.

**Overfitting-underfitting.py**: Script for evaluating the Random Forest model on the training set.

**FeatureEngineer.py**: Script for feature selection and further training of the Random Forest model.

**ML-Models.py**: Script for evaluating different machine learning models.

**Dataset**

The dataset includes raw process sensor data structured as matrices (tab-delimited), with rows representing the cycles and columns representing data points within a cycle. The sensors involved are:


PS2: Pressure (bar), sampled at 100 Hz
FS1: Volume flow (l/min), sampled at 10 Hz
The target condition values are cycle-wise annotated in profile.txt, which includes:

Cooler condition (%)

**Valve condition** (%)

Internal pump leakage

Hydraulic accumulator pressure (bar)

Stable flag

Scripts Overview

**data_analysis.py**

This script performs the following tasks:

Re-extracts the contents of the zip file.

Loads the profile data.

Performs feature engineering by calculating aggregated statistical features (mean, median, std) for each cycle.

Applies feature scaling.

Creates polynomial and interaction features.

Performs a stratified split with 2000 samples for training and the rest for testing.

Defines a parameter grid for SVM.

Performs hyperparameter tuning using GridSearchCV for SVM.

Evaluates the best SVM model on the test set and displays the results.

**RandomForest.py**

This script performs the following tasks:

Re-imports necessary libraries and re-extracts the contents of the zip file.

Loads the profile data.

Performs feature engineering and feature scaling.

Creates polynomial and interaction features.

Splits the data into training and testing sets with 2000 samples for training and the rest for testing.

Trains and evaluates a Random Forest model with the specified training and testing split.

Displays evaluation results for the Random Forest model.

**Overfitting-underfitting.py**

This script evaluates the Random Forest model on the training set and displays the training evaluation results.

**FeatureEngineer.py**

This script performs the following tasks:

Re-imports necessary libraries and re-extracts the contents of the zip file.

Loads the profile data.

Performs feature engineering and feature scaling.

Creates polynomial and interaction features.

Applies feature selection using RFE with Random Forest.

Splits the data into training and testing sets with 2000 samples for training and the rest for testing.

Trains and evaluates a Random Forest model with the selected features and stratified training samples.

Displays evaluation results for the enhanced Random Forest model.

**ML-Models.py**

This script evaluates the performance of different machine learning models (SVM, KNN, Logistic Regression, Decision Tree, AdaBoost) on the dataset:

Re-imports necessary libraries and re-extracts the contents of the zip file.

Loads the profile data.

Performs feature engineering and feature scaling.

Creates polynomial and interaction features.

Splits the data into training and testing sets with 2000 samples for training and the rest for testing.

Defines and evaluates each machine learning model.

Displays evaluation results for each model.

**Requirements**

To run the scripts, you will need the following Python libraries:

zipfile
pandas
numpy
sklearn
