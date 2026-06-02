import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import os  # Import os module for directory operations

"""
Support Vector Machine (SVM) can be applied to both classification (C) and regression (R) tasks. 
Due to the presence of multiple hyperparameters, grid search is commonly used to explore different parameter combinations. 
The most important hyperparameters include the kernel function and the penalty parameter C.

Overall workflow:
1. Load the dataset
2. Split the dataset into training and testing sets
3. Perform grid search (train the model and evaluate on the test set) to identify the optimal hyperparameters
4. Save the best test results to an Excel file
5. Load an additional test dataset, perform prediction using the best model, and save the results
"""
# 1. Load the main dataset
data = pd.read_excel("./spectral.xlsx", header=0)
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
# 2. Perform grid search to determine optimal hyperparameters
kernels = ['linear', 'rbf', 'sigmoid', 'poly']
x_count = 1
best_score = 0
best_kernel = None
best_c = None
for kernel in kernels:
    for c in range(1, 5):
        model = svm.SVC(kernel=kernel, C=c, probability=True)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_test)
        
        if score > best_score:
            best_score = score
            best_c = c
            best_kernel = kernel
        
        print(f'{x_count} iteration, kernel:{kernel}, C:{c} results:')
        print(score)
        print('-' * 50)
        x_count += 1
# Output the optimal hyperparameters
print(f"best_kernel={best_kernel}, best_c={best_c}, best_score={best_score}")
# 3. Train the optimal model and evaluate on the main test set
best_model = svm.SVC(kernel=best_kernel, C=best_c, probability=True)
best_model.fit(x_train, y_train)
y_predict = best_model.predict(x_test)
# Define result saving directory and create it if it does not exist
save_dir = './Test result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory {save_dir} has been created")
# Save prediction results of the main test set
y_test = np.array(y_test)
y_predict = np.array(y_predict)
data_save = pd.DataFrame({'true': y_test, 'predict': y_predict})
data_save.to_excel(os.path.join(save_dir, 'SVM_Test set prediction results.xlsx'))
# Output classification report for the main test set
report = classification_report(y_test, y_predict)
print("Best test set results:")
print(report)

print('='*30)

# 4. Load an additional test dataset and perform prediction
# Note: Replace "Test spectral.xlsx" with the actual file path if needed
data_new = pd.read_excel("Test spectral.xlsx", header=None)
new_x = data_new.iloc[:, 1:]
new_y = data_new.iloc[:, 0]

# Use the optimal model to predict the new dataset
new_y_predict = best_model.predict(new_x)

# 5. Save prediction results of the additional test dataset
new_y = np.array(new_y)
new_y_predict = np.array(new_y_predict)
new_data_save = pd.DataFrame({'true': new_y, 'predict': new_y_predict})

# Construct saving path and ensure directory exists
new_save_path = os.path.join(save_dir, 'SVM_Additional_test_set_results.xlsx')
new_data_save.to_excel(new_save_path)
print(f"Additional test set prediction results saved to: {new_save_path}")

# Output classification report for the additional test set
new_report = classification_report(new_y, new_y_predict)
print("Additional test set results:")
print(new_report)


import joblib

joblib.dump(best_model, './models/SVM_model.pkl')
print("SVM model saved")