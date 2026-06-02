import tensorflow as tf
from tensorflow import  keras
from  sklearn.metrics import  r2_score
import  numpy as np
import  pandas as pd
from  sklearn.model_selection import  train_test_split
"""
Framework: TensorFlow

Description:
This implementation uses TensorFlow to build a simple and easy-to-understand model.

For multi-class classification tasks (more than two classes), the labels y are converted to one-hot encoding, similar to BPNN. The main difference is the network architecture.

Since the feature dimension is only 9D and the number of features is small, a single convolution layer with a pooling layer is used to reduce the risk of overfitting.

Workflow:
1. Read the dataset
2. Split the dataset into training and testing sets
3. Build the network architecture and configure hyperparameters
4. Train the model
5. Evaluate the model on the test set
6. Save the best test results to an Excel file
"""

# 1. Load dataset
data = pd.read_excel("./spectral.xlsx", header=0)
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Preserve original class labels for mapping predictions back
classes = sorted(y.unique())

# Perform stratified sampling using original labels, then apply one-hot encoding
x_train, x_test, y_train_raw, y_test_raw = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

y_train = pd.get_dummies(y_train_raw).reindex(columns=classes, fill_value=0)
y_test = pd.get_dummies(y_test_raw).reindex(columns=classes, fill_value=0)

# Convert to numpy and reshape for CNN input format (samples, 8, 1)
x_train = np.array(x_train, dtype=np.float32).reshape(-1, 8, 1)
x_test = np.array(x_test, dtype=np.float32).reshape(-1, 8, 1)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# 2. Build CNN model
input = keras.Input(shape=(8, 1))
x = keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation='relu')(input)
x = keras.layers.MaxPool1D(pool_size=2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=32, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(units=8, activation='relu')(x)
x = keras.layers.Dense(units=3, activation='softmax')(x)

model = keras.Model(input, x)
model.summary()

# Compile model with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 3. Train the model
model.fit(x_train, y_train, epochs=600, batch_size=16)
# 4. Evaluate model on test set
score = model.evaluate(x_test, y_test)
print(f"loss={score[0]},acc={score[1]}")
# 5. Predict on test set
y_predict = model.predict(x_test)
# Convert probability outputs to class labels
y_test_label = tf.argmax(y_test, axis=1).numpy()
y_predict_label = tf.argmax(y_predict, axis=1).numpy()
# Map predicted indices back to original class labels
y_test_label = np.array([classes[i] for i in y_test_label])
y_predict_label = np.array([classes[i] for i in y_predict_label])
# Create directory for saving results if it does not exist
save_dir = './Test result'
if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
# Save test set prediction results
data_save = pd.DataFrame({'true': y_test_label, 'predict': y_predict_label})
data_save.to_excel('./Test result/CNN_Test set prediction results.xlsx')
# Calculate and print test accuracy
test_acc = np.mean(y_test_label == y_predict_label)
print(f"Test set accuracy={test_acc}")

print('=' * 30)
# 6. Load external test dataset
data_new = pd.read_excel("./Test spectral.xlsx", header=None)
new_x = data_new.iloc[:, 1:]
new_y = data_new.iloc[:, 0]
# Reshape external data to match CNN input format
new_x = np.array(new_x, dtype=np.float32).reshape(-1, 8, 1)
new_y = np.array(new_y)
# Predict on external dataset
new_y_predict = model.predict(new_x)
new_y_predict_label = tf.argmax(new_y_predict, axis=1).numpy()
# Map predictions back to original class labels
new_y_predict_label = np.array([classes[i] for i in new_y_predict_label])
# Save external test results
new_data_save = pd.DataFrame({'true': new_y, 'predict': new_y_predict_label})
new_data_save.to_excel('./Test result/CNN_Additional_test_set_results.xlsx')
print("External test results saved to: ./Test result/CNN_Additional_test_set_results.xlsx")
# Calculate and print external test accuracy
new_acc = np.mean(new_y == new_y_predict_label)
print(f"Additional test set accuracy={new_acc}")














import os

if os.path.exists('./models/CNN_model.keras'):
    os.remove('./models/CNN_model.keras')

model.save('./models/CNN_model.keras')
print("CNN model saved")