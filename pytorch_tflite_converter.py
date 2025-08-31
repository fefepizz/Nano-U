"""
PyTorch to TFLite Model Converter

This module provides functionality to convert PyTorch models to TensorFlow Lite format,
which is optimized for mobile and edge devices. The conversion process includes:
1. Loading a PyTorch model
2. Converting it to ONNX format
3. Converting ONNX to TensorFlow format
4. Applying dynamic quantization to reduce model size
5. Verifying the quantized model works correctly

Author: fefepizz
Date: August 31, 2025
"""

import os
import torch
import onnx
import tensorflow as tf
from models.Nano_U import Nano_U


def load_model(pth_path):
    """
    Load a PyTorch model from a .pth file.
    
    Args:
        pth_path (str): Path to the .pth model file
        
    Returns:
        torch.nn.Module: Loaded PyTorch model in evaluation mode
    """
    model = Nano_U(n_channels=3)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()
    return model


def export_to_onnx(model, onnx_path, input_shape=(1, 3, 64, 48)):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): PyTorch model to export
        onnx_path (str): Output path for the ONNX model
        input_shape (tuple): Input tensor shape (batch, channels, height, width)
    """
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, 
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True
    )


def convert_onnx_to_tflite_dynamic_quantization(onnx_path, tflite_path):
    """
    Convert an ONNX model to TFLite format with dynamic quantization.
    
    Args:
        onnx_path (str): Path to the ONNX model
        tflite_path (str): Output path for the TFLite model
    """
    import onnx2tf
    
    # Temporary directory for the TensorFlow model
    tf_model_path = "tf_model"
    
    # Convert ONNX to TensorFlow format
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
    )

    # Apply dynamic quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    

def verify_quantized_model(tflite_path, input_shape=(1, 3, 64, 48)):
    """
    Verify the quantized TFLite model works correctly by running inference.
    
    Args:
        tflite_path (str): Path to the TFLite model
        input_shape (tuple): Input tensor shape (batch, channels, height, width)
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create a test input tensor
    test_input = torch.randn(*input_shape).numpy().astype('float32')
    
    # Convert from PyTorch format (NCHW) to TensorFlow format (NHWC)
    if len(test_input.shape) == 4:
        test_input = test_input.transpose(0, 2, 3, 1)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Calculate and print model size
    model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    """
    Main execution function to demonstrate the PyTorch to TFLite conversion workflow.
    
    Steps:
    1. Load the PyTorch model from disk
    2. Convert to ONNX format
    3. Convert ONNX to TFLite with dynamic quantization
    4. Verify the quantized model works correctly
    """
    # Define file paths
    pth_model_path = "models/MU_Net_distilled.pth"
    onnx_model_path = "models/MU_Net.onnx"
    tflite_model_path = "models/MU_Net_quantized.tflite"
    
    # Step 1: Load the PyTorch model
    model = load_model(pth_model_path)
    
    # Step 2: Export to ONNX format
    export_to_onnx(model, onnx_model_path)
    
    # Step 3: Convert to TFLite with dynamic quantization
    convert_onnx_to_tflite_dynamic_quantization(onnx_model_path, tflite_model_path)
    
    # Step 4: Verify the quantized model
    verify_quantized_model(tflite_model_path)
    
    print(f"Conversion completed: {pth_model_path} → {tflite_model_path}")