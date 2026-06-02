import tensorflow as tf
from tensorflow import  keras
from  sklearn.metrics import  r2_score
import  numpy as np
import  pandas as pd
from  sklearn.model_selection import  train_test_split

"""
Structure:tensorflow
Difference: For tasks involving more than two categories, the labels y need to be one-hot encoded.

Overall workflow:
1. Read the dataset
2. Split the dataset into training and testing sets
3. Build the network architecture and configure hyperparameters
4. Train the model
5. Evaluate the model on the test set
6. Save the best test results to an Excel file
"""
# 1. Load dataset
data = pd.read_excel("./spectral.xlsx", header=0)
print("spectral.xlsx shape:", data.shape)
print(data.head())
x = data.iloc[:, 1:]
print("x shape:", x.shape)
y = data.iloc[:, 0]
# Perform stratified split using original labels, then apply one-hot encoding
x_train, x_test, y_train_raw, y_test_raw = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
# Convert labels to one-hot encoding for neural network training
y_train = pd.get_dummies(y_train_raw)
y_test = pd.get_dummies(y_test_raw)
# 2. Build neural network model
input = keras.Input(shape=(8,))
x = keras.layers.Dense(units=16, activation='relu')(input)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(units=8, activation='relu')(x)
x = keras.layers.Dense(units=3, activation='softmax')(x)

model = keras.Model(input, x)
# Compile model with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 3. Train the model
model.fit(x_train, y_train, epochs=500, batch_size=16)
# 4. Evaluate on test set
score = model.evaluate(x_test, y_test)
print(f"loss={score[0]},acc={score[1]}")
# 5. Predict on test set
y_predict = model.predict(x_test)
y_test = np.array(y_test)
y_predict = np.array(y_predict)
# Convert predictions from probability vectors to class labels
y_test = tf.argmax(y_test, axis=1).numpy()
y_predict = tf.argmax(y_predict, axis=1).numpy()
# Create directory for saving results if it does not exist
save_dir = './Test result'
if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
# Save test set prediction results
data_save = pd.DataFrame({'true': y_test, 'predict': y_predict})
data_save.to_excel('./Test result/BPNN_Test set prediction results.xlsx')
# Calculate and print test accuracy
test_acc = np.mean(y_test == y_predict)
print(f"Test set accuracy={test_acc}")

print('=' * 30)
# 6. Load external test dataset
data_new = pd.read_excel("./Test spectral.xlsx", header=None)
new_x = data_new.iloc[:, 1:]
new_y = data_new.iloc[:, 0]
# Predict on external dataset (probabilities -> class labels)
new_y_predict = model.predict(new_x)
new_y_predict = np.array(new_y_predict)
new_y_predict = tf.argmax(new_y_predict, axis=1).numpy()
# Save external test results
new_y = np.array(new_y)
new_data_save = pd.DataFrame({'true': new_y, 'predict': new_y_predict})
new_data_save.to_excel('./Test result/BPNN_Additional_test_set_results.xlsx')

print("External test results saved to：./Test result/BPNN_Additional_test_set_results.xlsx")
# Calculate and print external test accuracy
new_acc = np.mean(new_y == new_y_predict)
print(f"Additional test set accuracy={new_acc}")




















import os

if os.path.exists('./models/BPNN_model.keras'):
    os.remove('./models/BPNN_model.keras')

model.save('./models/BPNN_model.keras')
print("BPNN model saved")