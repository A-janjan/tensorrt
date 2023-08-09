from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import copy
import time
from tensorflow import keras
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants
import numpy as np



## convert to TRT

def load_with_converter(path, precision, batch_size):
    """Loads a saved model using a TF-TRT converter, and returns the converter
    """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    if precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
    elif precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
        precision_mode = trt.TrtPrecisionMode.FP32
    else:
        params = params._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
        maximum_cached_engines=100,
        minimum_segment_size=3,
        allow_build_at_runtime=True
        )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
    )

    return converter


SAVEDMODEL_PATH = 'saved_model'


converter = load_with_converter(
            os.path.join(SAVEDMODEL_PATH),
            precision='fp16',
            batch_size=512
        )


xx = converter.convert()

converter.save(
            os.path.join(SAVEDMODEL_PATH, "converted")
        )



SAVEDMODEL_PATH = 'saved_model'
trt_model_dir = os.path.join(SAVEDMODEL_PATH, "converted")
saved_model_loaded = tf.saved_model.load(trt_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)


load_name = 'figs_class_test.pickle'
with open(load_name, 'rb') as handle:
    x_test, y_test = pickle.load(handle)


################### measure time ##################################


# Start the timer
start_time = time.time()


def preprocess_input(x):
    # Normalize the input between -1 and 1
    x = (x / 255.0) * 2 - 1
    # Apply channel-wise mean subtraction
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    return x

x_test_pre = preprocess_input(x_test)

trt_predictions = graph_func(tf.convert_to_tensor(x_test_pre, dtype=tf.float32))

# End the timer
end_time = time.time()

# Calculate the execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000

# Print the execution time
print(f"Execution time: {execution_time_ms} milliseconds")

##################################################################

trt_predicts = []
for x in trt_predictions[0]:
  trt_predicts.append(x)


trt_preds = []
for x in trt_predicts:
  if(x>=0.5):
    trt_preds.append(1)
  else:
    trt_preds.append(0)

qz = 0
for i in range(749):
  if(y_test.tolist()[i]==trt_preds[i]):
    qz += 1
print(qz)


acc = qz/len(y_test)

print('accuracy = {}'.format(acc))



print(classification_report(y_test, trt_preds))
print(confusion_matrix(y_test, trt_preds))
