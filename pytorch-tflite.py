"""
This script converts a PyTorch model to a TFLite model using ai-edge-torch.

It loads a pre-trained Nano_U PyTorch model, converts it to TFLite format,
and verifies the conversion by comparing the output of the original and
converted models.
"""
# https://ai.google.dev/edge/litert/models/pytorch_to_tflite
# use ai-edge-torch conda env

import ai_edge_torch
import numpy as np
import torch
from models.Nano_U import Nano_U

# Load the pre-trained PyTorch model
model = Nano_U(n_channels=3)
model.load_state_dict(torch.load('models/Nano_U.pth', map_location=torch.device('cpu')))
model.eval()

# Create a sample input tensor for tracing the model
sample_input = (torch.randn(1, 3, 48, 64),)

# Get the output from the original PyTorch model
torch_output = model(*sample_input)

# Convert the PyTorch model to a TFLite model
edge_model = ai_edge_torch.convert(model, sample_input)

# Save the converted TFLite model
tflite_model_path = "models/Nano_U.tflite"
edge_model.export(tflite_model_path)

# Run inference with the converted TFLite model
edge_output = edge_model(*sample_input)

# Verify that the outputs of the two models are close
is_close = np.allclose(torch_output[0].detach().numpy(), edge_output[0], atol=1e-5)
print(f"The outputs of the PyTorch and TFLite models are close: {is_close}")

if not is_close:
    print("Verification failed: The outputs are not close enough.")
else:
    print("Verification successful!")
