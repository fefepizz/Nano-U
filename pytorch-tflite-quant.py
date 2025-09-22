"""
Convert a trained Nano-U PyTorch model to TensorFlow Lite (TFLite) format with dynamic range quantization.

This script optimizes the model for edge deployment by reducing its size and enabling faster inference,
with minimal loss in precision. It uses real validation images for quantization calibration.

Usage:
    pip install tf-nightly torch torchvision ai-edge-torch numpy opencv-python
    python pytorch-tflite-quant.py

Output:
    The quantized TFLite model is saved to models/provissima2.tflite
"""

import torch
import ai_edge_torch
import tensorflow as tf
import numpy as np
from models.Nano_U import Nano_U


class ModelWrapper(torch.nn.Module):
    """
    Wraps the Nano-U model to handle NHWC input/output for compatibility with TFLite conversion.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.model(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        return torch.reshape(x, (1, 48, 64, 1))


model = Nano_U(n_channels=3)
model.load_state_dict(torch.load('models/Nano_U_2L.pth'))
model.eval()
model = ModelWrapper(model)

sample_input = (torch.randn(1, 48, 64, 3),)


def load_validation_data():
    """
    Loads up to 100 validation images in NHWC format for quantization calibration.
    Returns:
        List of numpy arrays with shape (1, 48, 64, 3), dtype float32, normalized to [0, 1].
    """
    import os
    import cv2
    val_img_dir = "data/processed_data/val/img"
    val_img_files = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(".png")])
    print(f"Loading {len(val_img_files)} validation images from {val_img_dir}")
    validation_images = []
    for i, img_path in enumerate(val_img_files[:100]):
        img = cv2.imread(img_path)
        if img is not None:
            img = img[:, :, ::-1]  # BGR -> RGB
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            validation_images.append(img)
            if i == 0:
                print(f"First validation image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        else:
            print(f"Warning: Could not load image {img_path}")
    print(f"Successfully loaded {len(validation_images)} validation images for quantization")
    return validation_images


def representative_dataset_gen():
    """
    Generator for representative dataset used in TFLite quantization calibration.
    Yields:
        List containing one validation image in NHWC format.
    """
    print("Loading validation data for quantization calibration...")
    validation_images = load_validation_data()
    print(f"Using {len(validation_images)} validation images for representative dataset")
    for i, img in enumerate(validation_images):
        if i == 0:
            print(f"Representative dataset - First image shape: {img.shape}, dtype: {img.dtype}")
        yield [img]


tfl_converter_flags = {
    'optimizations': [tf.lite.Optimize.DEFAULT],
    'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]},
    'inference_input_type': tf.int8,
    'inference_output_type': tf.int8,
    'allow_custom_ops': True,
    'representative_dataset': representative_dataset_gen
}

tfl_drq_model = ai_edge_torch.convert(
    model, sample_input, _ai_edge_converter_flags=tfl_converter_flags
)

tfl_drq_model.export("models/Nano_U_int8.tflite")
print("Successfully converted and saved the quantized model to models/Nano_U_int8.tflite")