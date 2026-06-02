from sklearn.tree  import DecisionTreeClassifier
from  sklearn.model_selection  import  train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib
"""
Model: Decision Tree

Description:
This script implements a Decision Tree classifier for spectral data classification.
Grid search is applied to find the optimal hyperparameters.

Workflow:
1. Load the dataset from 'spectral.xlsx'
2. Split the dataset into training and testing sets (80:20 stratified sampling)
3. Perform grid search to determine optimal hyperparameters
4. Train the model with the best parameters
5. Evaluate the model on the test set
6. Save prediction results to an Excel file
7. Load an external dataset ('Test spectral.xlsx')
8. Perform prediction on the external dataset
9. Save external test results and evaluation metrics
"""

# 1. Load main dataset
data = pd.read_excel("./spectral.xlsx", header=0)
x = data.iloc[:, 1:]  # Feature matrix
y = data.iloc[:, 0]   # Class labels
# Split dataset into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
# 2. Perform grid search for optimal hyperparameters
criterions = ['entropy', 'gini']
max_depths = [50, 100, 150, 200]
x_count = 1
temp = 0
best_criterion = None
best_max_depth = None

for criterion in criterions:
    for max_depth in max_depths:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        model.fit(x_train, y_train)

        # Predict on test set
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_test)

        # Update best parameters
        if temp < score:
            temp = score
            best_criterion = criterion
            best_max_depth = max_depth

        print(f'{x_count}times,criterion:{criterion},max_depth:{max_depth}_results:')
        print(score)
        print('-' * 50)
        x_count += 1

# Output best hyperparameters
print(f"best_criterion={best_criterion},best_max_depth={best_max_depth}")
# 3. Train final model with best parameters
best_model = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_max_depth)
best_model.fit(x_train, y_train)
# 4. Predict on test set
y_predict = best_model.predict(x_test)
# Convert to numpy arrays
y_test = np.array(y_test)
y_predict = np.array(y_predict)
# Create directory for saving results (using TensorFlow file API)
import tensorflow as tf
save_dir = './Test result'
if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
# Save test set prediction results
data_save = pd.DataFrame({'true': y_test, 'predict': y_predict})
data_save.to_excel('./Test result/DT_Test set prediction results.xlsx')
# Generate classification report for test set
report = classification_report(y_test, y_predict)
print("Best test set results:")
print(report)

print('=' * 30)

# 5. Load external test dataset
data_new = pd.read_excel("./Test spectral.xlsx", header=None)
new_x = data_new.iloc[:, 1:]
new_y = data_new.iloc[:, 0]
# Predict on external dataset
new_y_predict = best_model.predict(new_x)
# Convert to numpy arrays
new_y = np.array(new_y)
new_y_predict = np.array(new_y_predict)
# Save external test results
new_data_save = pd.DataFrame({'true': new_y, 'predict': new_y_predict})
new_data_save.to_excel('./Test result/DT_Additional_test_set_results.xlsx')
print("External test results saved to: ./Test result/DT_Additional_test_set_results.xlsx")
# Generate classification report for external dataset
new_report = classification_report(new_y, new_y_predict)
print("Additional test set results:")
print(new_report)



joblib.dump(best_model, './models/DT_model.pkl')
print("DT model saved")