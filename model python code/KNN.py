import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os

"""
Model: K-Nearest Neighbors (KNN)

Description:
This script implements a KNN classifier for spectral data classification.
The feature scales are relatively similar, so no normalization or standardization is required.

Since the dataset is relatively small, an 80:20 stratified sampling strategy is used 
to split the data into training and testing sets.

Workflow:
1. Load the dataset from 'spectral.xlsx'
2. Split the dataset into training and testing sets and train the model
4. Evaluate the model on the test set and save the prediction results to an Excel file
6. Load an external dataset ('Test spectral.xlsx')
7. Perform prediction on the external dataset
8. Save external test results and evaluation metrics
"""
# 1. Load main dataset
data = pd.read_excel("./spectral.xlsx", header=0)
x = data.iloc[:, 1:]  # Feature matrix
y = data.iloc[:, 0]   # Class labels
# Split dataset into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
# 2. Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
# 3. Predict on test set
y_predict = model.predict(x_test)
# Convert to numpy arrays
y_test = np.array(y_test)
y_predict = np.array(y_predict)
# Create directory for saving results if it does not exist
save_dir = './Test result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 4. Save test set prediction results
data_save = pd.DataFrame({'true': y_test, 'predict': y_predict})
data_save.to_excel('./Test result/KNN_Test set prediction results.xlsx')
# 5. Generate classification report for test set
report = classification_report(y_test, y_predict)
print("Best test set results:")
print(report)
print('=' * 30)
# 6. Load external test dataset
data_new = pd.read_excel("./Test spectral.xlsx", header=None)
new_x = data_new.iloc[:, 1:]
new_y = data_new.iloc[:, 0]
# Predict on external dataset
new_y_predict = model.predict(new_x)
# Convert to numpy arrays
new_y = np.array(new_y)
new_y_predict = np.array(new_y_predict)
# 7. Save external test results
new_data_save = pd.DataFrame({'true': new_y, 'predict': new_y_predict})
new_data_save.to_excel('./Test result/KNN_Additional_test_set_results.xlsx')
print("External test results saved to: ./Test result/KNN_Additional_test_set_results.xlsx")
# 8. Generate classification report for external dataset
new_report = classification_report(new_y, new_y_predict)
print("Additional test set results:")
print(new_report)










import joblib

joblib.dump(model, './models/KNN_model.pkl')
print("KNN model saved")