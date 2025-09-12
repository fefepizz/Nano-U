"""
This script converts a trained Nano-U PyTorch model to a TensorFlow Lite (TFLite)
format with dynamic range quantization. This process optimizes the model for
deployment on edge devices by reducing its size and enabling faster inference,
with a slight trade-off in precision.

Requirements:
- Python 3.10
- TensorFlow (`tf-nightly`)
- PyTorch (`torch`, `torchvision`)
- AI Edge Torch (`ai-edge-torch`)

Installation:
    pip install tf-nightly torch torchvision ai-edge-torch numpy
"""
# https://ai.google.dev/edge/litert/models/pytorch_to_tflite
# python=3.10
# linux
# pip install tf-nightly torch==2.6.0 torchvision==0.21.0
# pip install ai-edge-torch

import torch
import ai_edge_torch
import tensorflow as tf
import numpy as np
from models.Nano_U import Nano_U


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Permute input from NHWC to NCHW
        x = x.permute(0, 3, 1, 2)
        # Run the model
        x = self.model(x)
        # Permute output from NCHW to NHWC
        x = x.permute(0, 2, 3, 1)
        # Reshape the output to the desired 4D shape
        return torch.reshape(x, (1, 48, 64, 1))


# The model is loaded from a saved state dictionary and set to evaluation mode.
model = Nano_U(n_channels=3)
model.load_state_dict(torch.load('models/Nano_U_2L.pth'))
# No longer need channels_last here as we handle it in the wrapper
model.eval()

# Wrap the model to handle NHWC conversion and output squeezing
model = ModelWrapper(model)

# 2. Define a sample input for the model in NHWC format
# This is required by the converter to trace the model's architecture.
sample_input = (torch.randn(1, 48, 64, 3),)


def representative_dataset_gen():
    """
    Generator function for the representative dataset.
    This dataset is used by the TFLite converter to calibrate the quantization
    parameters for the model's weights and activations. It should yield a
    list of input tensors that are representative of the data the model will
    see during inference.
    """
    for _ in range(100):
        # Create a random input tensor with the NHWC shape.
        yield [np.random.rand(1, 48, 64, 3).astype(np.float32)]


# These flags control the optimization and conversion process.
tfl_converter_flags = {
    
    # Enable default optimizations, which include quantization.
    'optimizations': [tf.lite.Optimize.DEFAULT],
    
    # quantized operations.
    'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]},
    
    # Set the expected input and output types for the quantized model.
    'inference_input_type': tf.int8,
    'inference_output_type': tf.int8,
    
    # Allow custom operations if any are encountered during conversion.
    'allow_custom_ops': True,
    
    # Provide the representative dataset for calibration.
    'representative_dataset': representative_dataset_gen
}

# The `ai_edge_torch.convert` function takes the model, a sample input, and
# the conversion flags to produce a quantized TFLite model.
tfl_drq_model = ai_edge_torch.convert(
    model, sample_input, _ai_edge_converter_flags=tfl_converter_flags
)


tfl_drq_model.export("models/provissima2.tflite")
print("Successfully converted and saved the quantized model to models/provissima2.tflite")