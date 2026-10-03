import numpy as np
import tensorflow as tf
import tf2onnx
import torch
import onnx
from onnx2torch import convert

# ==========================================
# 1. Paths Configuration
# ==========================================
H5_MODEL_PATH = "models/unet_lung_seg.hdf5"
ONNX_MODEL_PATH = "models/unet_lung_seg.onnx"
PYTORCH_WEIGHTS_PATH = "models/unet_lung_seg.pth"
PYTORCH_FULL_MODEL_PATH = "models/unet_lung_seg.pt"

# ==========================================
# 2. Load Keras HDF5 Model & Export to ONNX
# ==========================================
print("Loading TensorFlow model...")
tf_model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)

# Automatically extract the input signature from the Keras model
input_shape = tf_model.inputs[0].shape
input_dtype = tf_model.inputs[0].dtype

input_spec = (
    tf.TensorSpec(
        shape=[None if dim is None else dim for dim in input_shape],
        dtype=input_dtype,
        name=tf_model.inputs[0].name.split(":")[0]
    ),
)

print("Converting Keras to ONNX graph directly in memory...")

# 3. Convert without passing output_path (avoids Windows temporary file lock)
onnx_proto, _ = tf2onnx.convert.from_keras(
    tf_model,
    input_signature=input_spec,
    opset=13
)

# 4. Save ONNX file cleanly using onnx.save_model
onnx.save_model(onnx_proto, ONNX_MODEL_PATH)
print(f"Saved ONNX model successfully to '{ONNX_MODEL_PATH}'")

# 5. Convert to PyTorch directly using the saved path or proto
torch_model = convert(onnx_proto)
torch_model.eval()
print("Converted successfully to PyTorch Module!")

# ==========================================
# 4. Verify Numerical Equivalence
# ==========================================
print("Verifying numerical consistency between TF and PyTorch...")

# Generate a single synthetic sample (Batch size = 1)
test_dims = [1 if dim is None else dim for dim in input_shape]
dummy_input_np = np.random.randn(*test_dims).astype(np.float32)

# Run TensorFlow inference
tf_output = tf_model(dummy_input_np).numpy()

# Run PyTorch inference
torch_input = torch.from_numpy(dummy_input_np)
with torch.no_grad():
    torch_output = torch_model(torch_input).cpu().numpy()

# Compare outputs
max_abs_diff = np.max(np.abs(tf_output - torch_output))
print(f"Max absolute discrepancy: {max_abs_diff:.6e}")

assert max_abs_diff < 1e-4, "Discrepancy too high! Check for unaligned layers/activations."
print("Equivalence verified successfully!")

# ==========================================
# 5. Save the PyTorch Model
# ==========================================
# Save the complete module structure and weights
torch.save(torch_model, PYTORCH_FULL_MODEL_PATH)

# Save only the state_dict (standard PyTorch practice)
torch.save(torch_model.state_dict(), PYTORCH_WEIGHTS_PATH)

print(f"PyTorch model saved to '{PYTORCH_FULL_MODEL_PATH}' and '{PYTORCH_WEIGHTS_PATH}'")