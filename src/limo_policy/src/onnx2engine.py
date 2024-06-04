import os
import tensorrt as trt

# Get current working directory
current_path = os.getcwd()

# Initialize TensorRT logger with warning level
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# Define the path to the ONNX model
onnx_model_path = os.path.join(current_path, "model", "model.onnx")

# Create a builder and a network definition
builder = trt.Builder(TRT_LOGGER)
# Explicitly set the network to use explicit batch
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags=network_flags)
# Create an ONNX parser to load the ONNX model into the TensorRT network
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load the ONNX model into the TensorRT network
with open(onnx_model_path, 'rb') as model:
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# Create builder config
config = builder.create_builder_config()
# Allocate workspace size for the builder
config.max_workspace_size = 1 << 30  # 1GB

# Enable FP16 precision if supported and desired (uncomment if needed)
# if builder.platform_has_fast_fp16:
#     config.set_flag(trt.BuilderFlag.FP16)

# Build the engine
engine = builder.build_engine(network, config)
if engine is None:
    print("Failed to build the engine.")
    exit()

# Serialize the engine to a file
engine_file_path = os.path.join(current_path, "model", "model.engine")
with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())
print("Engine file has been built.")
