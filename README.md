**Data Preparation and Preprocessing**

**Loading Data:**


The code loads three data files: FS1.txt, PS2.txt, and profile.txt. Each file is read into a pandas DataFrame, with profile.txt having custom column names.
Reducing PS2 Dimensions:

To simplify the PS2 dataset, the mean of every 10 consecutive columns is calculated, reducing the number of columns to 600.
Creating Target Variable:

The target variable (target) is created based on the 'Optimal' column in the profile_df. It is set to 1 if the value is 100 and 0 otherwise.

**Scaling Data:**


The FS1 and PS2 data are scaled using MinMaxScaler to normalize the data between 0 and 1.

**Combining Data:**


The scaled FS1 and PS2 datasets are concatenated horizontally to form a combined dataset for time series modeling.
Time Series Dataset Creation

**Creating Time Series Dataset:**


A function is defined to create a time series dataset. It generates sequences of data points and corresponding targets for a specified number of time steps.

**Setting Time Steps and Splitting Data:**


The code sets the number of time steps to 10 and splits the dataset into training (first 2000 points) and testing sets (remaining points).


**LSTM Model**

Defining LSTM Model:

An LSTM model is defined with two LSTM layers followed by two Dense layers. The final Dense layer has a sigmoid activation for binary classification.
Compiling and Training LSTM Model:

The LSTM model is compiled with the Adam optimizer and binary cross-entropy loss function. It is then trained on the time series training data, with validation on the testing data.
Saving LSTM Model:

The trained LSTM model is saved to disk using Keras's model.save() function.

**Plotting Learning Curves**

Plotting Learning Curves:

A function is defined to plot the learning curves of the LSTM model, showing both accuracy and loss for training and validation sets over epochs.
Evaluating LSTM Model
Evaluating LSTM Model:
The LSTM model is evaluated on the testing data, and performance metrics are calculated and printed.

**Another Models Preparation**
 

For models like Logistic Regression, Random Forest, and SVM, the time series data is flattened since these models do not natively handle 3D input.

**Initializing Models:**

Three models are initialized: Logistic Regression, Random Forest, and SVM. These models will be trained and evaluated.

**Training and Evaluation**

Training and Evaluating Models:

Each model is trained on the flattened training data. Predictions are made on the testing data, and various performance metrics (accuracy, precision, recall, F1 score, and ROC-AUC) are calculated and printed.

**Saving Models:**

The trained models are saved to disk using pickle for future use.

**Model Comparison**

Comparing Models:
The performance metrics of all models (Logistic Regression, Random Forest, SVM, and LSTM) are compiled into a DataFrame and printed for comparison.


The saved models are loaded from disk, and predictions are made on the example testing data point. The predictions from each model are printed to verify they work as expected.
