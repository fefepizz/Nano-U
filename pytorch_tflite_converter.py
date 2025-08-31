import torch
import os
from models.MU_Net import MU_Net
import onnx
import tensorflow as tf

def load_model(pth_path):
    model = MU_Net(n_channels=3)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()
    return model

def export_to_onnx(model, onnx_path, input_shape=(1, 3, 64, 48)):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, 
        dummy_input,            # check this whole function 
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Exported to ONNX: {onnx_path}")

def convert_onnx_to_tflite_dynamic_quantization(onnx_path, tflite_path):
    
    import onnx2tf
    
    tf_model_path = "tf_model"
    
    onnx2tf.convert(            # controllare i parametri
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
    )

    # si può utilizzare anche la libreria di github

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    #check this
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    #check also this
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    
def verify_quantized_model(tflite_path, input_shape=(1, 3, 64, 48)):
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # sono tutte cose utili e necessarie?

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n=== Model Details ===")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Test inference with float32 input (dynamic quantization keeps input/output as float32)
    test_input = torch.randn(*input_shape).numpy().astype('float32')
    
    # Convert from NCHW (PyTorch format) to NHWC (TensorFlow format)
    if len(test_input.shape) == 4:  # Batch, Channel, Height, Width -> Batch, Height, Width, Channel
        test_input = test_input.transpose(0, 2, 3, 1)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Test inference successful")
    print(f"Output shape: {output_data.shape}")
    
    # Model size
    model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    pth_model_path = "models/MU_Net_distilled.pth"
    onnx_model_path = "models/MU_Net.onnx"
    tflite_model_path = "models/MU_Net_quantized.tflite"
    
    model = load_model(pth_model_path)
    
    export_to_onnx(model, onnx_model_path)
    
    convert_onnx_to_tflite_dynamic_quantization(onnx_model_path, tflite_model_path)
    
    verify_quantized_model(tflite_model_path)