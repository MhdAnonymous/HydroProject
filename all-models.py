import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

# Load your data files
fs1_df = pd.read_csv('FS1.txt', sep='\t', header=None)
ps2_df = pd.read_csv('PS2.txt', sep='\t', header=None)
profile_df = pd.read_csv('profile.txt', sep='\t', header=None, names=['Col1', 'Optimal', 'Col3', 'Col4', 'Condition'])

# Reducing PS2 to 600 columns by calculating the mean of every 10 consecutive columns
ps2_reduced = ps2_df.groupby(np.arange(ps2_df.shape[1]) // 10, axis=1).mean()

# Update the condition to be 1 if the second column value is 100, else 0
valve_condition = profile_df['Optimal']
target = (valve_condition == 100).astype(int).values

# Scale the data
scaler = MinMaxScaler()
fs1_scaled = scaler.fit_transform(fs1_df)
ps2_scaled = scaler.fit_transform(ps2_reduced)

# Prepare the combined dataset for time series modeling
combined_data = np.concatenate((fs1_scaled, ps2_scaled), axis=1)

# Create the time series dataset for LSTM
def create_time_series_dataset(data, target, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        Y.append(target[i + time_steps])
    return np.array(X), np.array(Y)

time_steps = 10  # Example time step, can be adjusted
X, y = create_time_series_dataset(combined_data, target, time_steps)

# Use the first 2000 data points for training and the rest for testing
train_size = 2000
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Prepare data for other models (flattened time series)
X_lr_train = X_train.reshape(X_train.shape[0], -1)
X_lr_test = X_test.reshape(X_test.shape[0], -1)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_lr_train, y_train)
    y_pred = model.predict(X_lr_test)
    y_pred_prob = model.predict_proba(X_lr_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

    print(f'{name} - Accuracy: {accuracy}')
    print(f'{name} - Precision: {precision}')
    print(f'{name} - Recall: {recall}')
    print(f'{name} - F1 Score: {f1}')
    print(f'{name} - ROC-AUC: {roc_auc}')
    print()

    # Save the model
    with open(f'{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model, f)

# LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and store history
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))

# Save the LSTM model
model.save('lstm_model.h5')

# Plot the learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

plot_learning_curves(history)

# Evaluate the LSTM model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM - Accuracy: {accuracy}')

# Predicting for additional metrics calculation
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate additional metrics for LSTM
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

results['LSTM'] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC-AUC': roc_auc
}

print(f'LSTM - Precision: {precision}')
print(f'LSTM - Recall: {recall}')
print(f'LSTM - F1 Score: {f1}')
print(f'LSTM - ROC-AUC: {roc_auc}')

# Compare models
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Example of testing data
example_data = X_test[:1]
example_target = y_test[:1]

# Load and test saved models
for name in models.keys():
    with open(f'{name.replace(" ", "_").lower()}.pkl', 'rb') as f:
        model = pickle.load(f)
        y_example_pred = model.predict(example_data.reshape(1, -1))
        print(f'{name} example prediction: {y_example_pred}')

# Load and test LSTM model
lstm_model = load_model('lstm_model.h5')
y_example_pred = (lstm_model.predict(example_data) > 0.5).astype(int)
print(f'LSTM example prediction: {y_example_pred}')

