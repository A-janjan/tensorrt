import cv2
import pickle
import numpy as np
import tensorflow as tf



with open('figs_class_test.pickle', 'rb') as handle:
    x_test, y_test = pickle.load(handle)


# load json and create model
json_file = open('model_fig_type.json', 'r')
model_json = json_file.read()
json_file.close()
classifier = tf.keras.models.model_from_json(model_json)

# Load the model weights from the H5 file
classifier.load_weights('model_fig_type.h5')
# load weights into new model
print("Loaded classifier")


shape = (80, 80, 3)
x_test_old = x_test.copy()
x_test = np.zeros((x_test.shape[0], *shape))
for i, x in enumerate(x_test_old):
    x_test[i] = cv2.resize(x, shape[:2])



def preprocess_input(x):
    # Normalize the input between -1 and 1
    x = (x / 255.0) * 2 - 1
    # Apply channel-wise mean subtraction
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    return x


import time

# Start the timer
start_time = time.time()


x_test_pre = preprocess_input(x_test)
predictions = classifier.predict(x_test_pre)

# End the timer
end_time = time.time()

# Calculate the execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000

# Print the execution time
print(f"Execution time: {execution_time_ms} milliseconds")


predictions = predictions[:, 0] > 0.5
acc = np.mean(predictions == y_test)
print('accuracy = {}'.format(acc))


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
