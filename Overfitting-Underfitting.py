# Evaluate the Random Forest model on the training set
y_train_pred_rf_2000_strat = rf_model_2000_strat.predict(X_train_2000_strat)

accuracy_train_rf_2000_strat = accuracy_score(y_train_2000_strat, y_train_pred_rf_2000_strat)
precision_train_rf_2000_strat = precision_score(y_train_2000_strat, y_train_pred_rf_2000_strat)
recall_train_rf_2000_strat = recall_score(y_train_2000_strat, y_train_pred_rf_2000_strat)
f1_train_rf_2000_strat = f1_score(y_train_2000_strat, y_train_pred_rf_2000_strat)

# Display training evaluation results
training_evaluation_results_rf_2000_strat = {
    'Accuracy': accuracy_train_rf_2000_strat,
    'Precision': precision_train_rf_2000_strat,
    'Recall': recall_train_rf_2000_strat,
    'F1 Score': f1_train_rf_2000_strat
}

training_evaluation_results_rf_2000_strat, rf_evaluation_results_2000_strat
