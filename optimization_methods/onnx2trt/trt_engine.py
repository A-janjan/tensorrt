import tensorrt as trt


class ModelData(object):
    MODEL_PATH = "model.onnx"
    INPUT_SHAPE = (3, 80, 80)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    # Enable explicit batch mode
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = 1 << 30

    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_serialized_network(network, config)


onnx_model_file = ModelData.MODEL_PATH

# Build a TensorRT engine.
serialized_engine = build_engine_onnx(onnx_model_file)

# Save the engine to a file for future use
with open("sample.engine", "wb") as f:
    f.write(serialized_engine)


# Contexts are used to perform inference.
# context = serialized_engine.create_execution_context()
# ...
