""""
goals: 
    - Load model from h5 and json files.(this model will be tensorflow model)
    - Do prediction on x_test data and then measure inference time and calculate accuracy

-> our work is about image processing.

"""

import cv2
import pickle
import numpy as np
import tensorflow as tf




dir = 'files/'
load_name = dir + 'figs_class_test.pickle'
with open(load_name, 'rb') as handle:
    x_test, y_test = pickle.load(handle)


# load json and create model
json_file = open(dir + 'model_fig_type.json', 'r')
model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(model_json)

# Load the model weights from the H5 file
model.load_weights(dir + 'model_fig_type.h5')
# load weights into new model
print("Loaded classifier")


shape = (80, 80, 3)
x_test_old = x_test.copy()
x_test = np.zeros((x_test.shape[0], *shape))
for i, x in enumerate(x_test_old):
    x_test[i] = cv2.resize(x, shape[:2])



from keras.applications.densenet import preprocess_input


import time

x_test_pre = preprocess_input(x_test)
x_test_pre = tf.convert_to_tensor(x_test_pre, dtype=tf.float32)

MAX_BATCH_SIZE=128
infer_batch_size = MAX_BATCH_SIZE // 2

list_x = []

# Start the timer

start = time.time()


# number of test data is 749
# so 749 / 64 ~= 11
for i in range(11):
   print(f"Step: {i}")
   start_idx = i * infer_batch_size
   end_idx   = (i + 1) * infer_batch_size
   x = x_test_pre[start_idx:end_idx, :]
   list_x.append(model.predict(x))

end = time.time()

# list_x would be (11, 64)


inference_time = end - start

print(f"================> inference time: {inference_time*1000}ms <=========================")


# make matrix of 0 and 1 for comparing with labels
preds = np.zeros((11,64))
for i in range(11):
	for j in range(64):
		if(list_x[i][j]>=0.5):
			preds[i][j]=1
		else:
			preds[i][j]=0


# calculate accuracy
# preds is a (11, 64) matrix
# but y_test is (749,)
# so

y_predicts = []
for i in range(11):
  for j in range(64):
    y_predicts.append(preds[i][j])


qz = 0
for i in range(len(y_predicts)):
  if(y_test[i]==y_predicts[i]):
    qz += 1
print(qz)


print(f"====> accuracy is {qz/len(y_predicts)}")



from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test[:len(y_predicts)], y_predicts))
print(confusion_matrix(y_test[:len(y_predicts)], y_predicts))