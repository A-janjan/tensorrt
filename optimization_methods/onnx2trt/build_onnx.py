from tensorflow import keras



# load json and create model
json_file = open('saved_model/model_fig_type.json', 'r')
model_json = json_file.read()
json_file.close()
classifier = keras.models.model_from_json(model_json)

# Load the model weights from the H5 file
classifier.load_weights('saved_model/model_fig_type.h5')
# load weights into new model
print("Loaded classifier")

################# Notice ############################3
#####   
###         pip install onnx
###         pip install tf2onnx
#####
############ convert to onnx #################################
import tf2onnx
import onnx

onnx_model, _ = tf2onnx.convert.from_keras(classifier)

onnx.save(onnx_model, "model.onnx")