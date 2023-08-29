####################################################
###### NOTICE: before applying following scripts
####### install tensorrt using 
################ !sudo apt-get install tensorrt
###################################################


import tensorflow.compiler as tf_cc
import cv2
import pickle
import numpy as np
import tensorflow as tf
import time
from keras.applications.densenet import preprocess_input
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants
import os


print("################ checking GPU and TensorRT ###############")

tf_cc.tf2tensorrt._pywrap_py_utils.is_tensorrt_enabled()
print()
tf.test.is_gpu_available()

print("############## LOAD MODEL AND DATA ####################")

load_name = 'figs_class_test.pickle'
with open(load_name, 'rb') as handle:
    x_test, y_test = pickle.load(handle)


# load json and create model
json_file = open('model_fig_type.json', 'r')
model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(model_json)

# Load the model weights from the H5 file
model.load_weights('model_fig_type.h5')
# load weights into new model
print("==============> Loaded MODEL <=======================")


print("##################### Preprocessing ######################")
shape = (80, 80, 3)
x_test_old = x_test.copy()
x_test = np.zeros((x_test.shape[0], *shape))
for i, x in enumerate(x_test_old):
    x_test[i] = cv2.resize(x, shape[:2])

x_test_pre = preprocess_input(x_test)
x_test_pre = tf.convert_to_tensor(x_test_pre, dtype=tf.float32)


SAVED_MODEL_DIR="saved_model"
model.save(SAVED_MODEL_DIR)



conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR,
    conversion_params=conversion_params)

converter.convert()

converter.save('saved_model/converted')


SAVEDMODEL_PATH = 'saved_model'
trt_model_dir = os.path.join(SAVEDMODEL_PATH, "converted")
saved_model_loaded = tf.saved_model.load(trt_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

import time

start = time.time()
trt_preds = graph_func(x_test_pre) # Use this to perform inference
end = time.time()

measured_time = (end-start)*1000

print(f"=============> infernece time : {measured_time}ms <========================")
