import os
import time
import torch
import numpy as np
import cv2
import gc
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from models.BU_Net.BU_Net_model import BU_Net
from models.Nano_U.Nano_U_model import Nano_U
from utils.LoadDataset import LoadDataset

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model_parameters(model_info):
    if model_info['type'] == 'pytorch':
        return sum(p.numel() for p in model_info['model'].parameters() if p.requires_grad)
    return model_info.get('pytorch_params', 0)

def get_model_size(model_path):
    return os.path.getsize(model_path) / 1024

def _run_tflite_inference(interpreter, image, target=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    
    img_np = image.cpu().numpy()
    input_data = np.zeros(expected_shape, dtype=np.float32)
    
    # Convert NCHW to NHWC
    for b in range(min(img_np.shape[0], expected_shape[0])):
        for c in range(min(img_np.shape[1], expected_shape[3])):
            resized = cv2.resize(img_np[b, c], (expected_shape[2], expected_shape[1]), cv2.INTER_AREA)
            input_data[b, :, :, c] = resized
    
    # Handle quantization
    if input_details[0]['dtype'] in [np.uint8, np.int8]:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = np.round(input_data / input_scale + input_zero_point).astype(input_details[0]['dtype'])
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if target is None:
        return output_data
    
    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    # Resize to match target
    output_processed = np.zeros((target.shape[0], target.shape[1], target.shape[2], target.shape[3]), dtype=np.float32)
    
    if len(output_data.shape) == 4:  # NHWC
        for b in range(min(output_data.shape[0], target.shape[0])):
            for c in range(min(output_data.shape[3], target.shape[1])):
                resized = cv2.resize(output_data[b, :, :, c], (target.shape[3], target.shape[2]), cv2.INTER_LINEAR)
                output_processed[b, c] = resized
    else:  # NHW
        for b in range(min(output_data.shape[0], target.shape[0])):
            resized = cv2.resize(output_data[b], (target.shape[3], target.shape[2]), cv2.INTER_LINEAR)
            output_processed[b, 0] = resized
    
    return torch.from_numpy(output_processed).to(target.device)

def measure_inference_time(model_info, dataloader, device, num_warmup=5):
    times = []
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device).eval()
    
    # Warmup
    for _ in range(num_warmup):
        image, _ = next(iter(dataloader))
        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(image.to(device))
        else:
            _run_tflite_inference(model, image)
    
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to(device)
            start = time.time()
            
            if model_type == 'pytorch':
                _ = model(image)
            else:
                _run_tflite_inference(model, image)
                
            times.append(time.time() - start)
    
    return times

def measure_accuracy(model_info, dataloader, device):
    ious, ce_accuracies = [], []
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device).eval()
    
    with torch.no_grad():
        for image, target in dataloader:
            image, target = image.to(device), target.to(device)
            
            if model_type == 'pytorch':
                output = model(image)
            else:
                output = _run_tflite_inference(model, image, target)
            
            # IoU
            pred = (output > 0.5).float()
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            ious.append(((intersection + 1e-6) / (union + 1e-6)).item())
            
            # Cross entropy accuracy
            if output.max() > 1.0 or output.min() < 0.0:
                pred_probs = torch.sigmoid(output)
            else:
                pred_probs = output
            
            target_binary = (target > 0.5).float()
            pred_probs = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
            
            ce_loss = -(target_binary * torch.log(pred_probs) + (1 - target_binary) * torch.log(1 - pred_probs))
            mean_ce_loss = ce_loss.mean().item()
            ce_accuracies.append(max(0, 1 - (mean_ce_loss / np.log(2))))
    
    return ious, ce_accuracies

def benchmark_model(model_info, dataloader, device):
    print(f"Benchmarking {model_info.get('name', 'Unknown')}...")
    
    times = measure_inference_time(model_info, dataloader, device)
    ious, ce_accuracies = measure_accuracy(model_info, dataloader, device)
    
    return {
        "avg_time_ms": np.mean(times) * 1000,
        "avg_iou": np.mean(ious),
        "avg_ce_accuracy": np.mean(ce_accuracies),
        "size_kb": get_model_size(model_info['path']),
        "params_m": get_model_parameters(model_info) / 1e6
    }

def get_single_prediction(model_info, image, target, device):
    """Get prediction from a single model"""
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device).eval()
        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            return output.cpu().numpy()
    else:
        output = _run_tflite_inference(model, image, target)
        # TFLite inference returns a numpy array directly when target is None
        if isinstance(output, np.ndarray):
            return output
        return output.cpu().numpy()

def process_frame(target_frame, target_mask, models, device, data_dir):
    """Process a single frame and return the figure"""
    # Load specific frame
    img_path = os.path.join(data_dir, "img", target_frame)
    mask_path = os.path.join(data_dir, "mask", target_mask)
    
    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        print(f"Target frame {target_frame} not found!")
        return None
    
    # Create dataset with single image
    dataset = LoadDataset([img_path], [mask_path])
    image, target = dataset[0]
    image = image.unsqueeze(0)  # Add batch dimension
    target = target.unsqueeze(0)  # Add batch dimension
    
    # Load original image for display
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load ground truth mask for comparison
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = (ground_truth > 127).astype(np.float32)  # Convert to binary
    
    # Get predictions from all models
    predictions = {}
    for name, model_info in models.items():
        print(f"Running inference with {model_info['name']} on {target_frame}...")
        pred = get_single_prediction(model_info, image, target, device)
        # Convert to binary mask and squeeze to remove batch/channel dimensions
        pred_binary = (pred > 0.5).astype(np.float32)
        if len(pred_binary.shape) == 4:
            pred_binary = pred_binary[0, 0]  # Remove batch and channel dims
        elif len(pred_binary.shape) == 3:
            pred_binary = pred_binary[0]  # Remove batch dim
        
        # Resize prediction to match ground truth size
        if pred_binary.shape != ground_truth.shape:
            pred_binary = cv2.resize(pred_binary, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        predictions[model_info['name']] = pred_binary

    def create_colored_overlay(gt_mask, pred_mask):
        """Create colored overlay based on the template from utils/metrics.py"""
        # Ensure masks are binary
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Create overlay: yellow for GT only, red for prediction only, green for overlap
        # Following the same color scheme as in utils/metrics.py plot_prediction function
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        overlay[(gt_mask == 1) & (pred_mask == 0)] = [1, 1, 0]  # Yellow: GT only (False Negative)
        overlay[(gt_mask == 0) & (pred_mask == 1)] = [1, 0, 0]  # Red: Prediction only (False Positive)
        overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 1, 0]  # Green: Overlap (True Positive)
        
        return overlay

    # Create matplotlib figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Model Predictions for {target_frame}', fontsize=16, fontweight='bold')
    
    # Original image spans middle column of top row
    axes[0, 1].imshow(original_img)
    axes[0, 1].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hide unused top subplots
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    
    # Model predictions in bottom row with colored overlays
    model_names = list(predictions.keys())
    for i, name in enumerate(model_names):
        if i < 3:  # Ensure we don't exceed the number of available subplots
            overlay = create_colored_overlay(ground_truth, predictions[name])
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'{name}\n(Yellow=GT only, Red=Pred only, Green=Overlap)', fontsize=10, fontweight='bold')
            axes[1, i].axis('off')
    
    return fig

def create_tflite_microflow_comparison(tflite_model_info, device, data_dir, microflow_prediction_path="../microflow_test/output/prediction_frame2.png"):
    """Create a comparison figure between TFLite model and microflow implementation"""
    target_frame = "frame2.png"
    
    # Check if microflow prediction exists
    if not os.path.exists(microflow_prediction_path):
        print(f"Microflow prediction not found at {microflow_prediction_path}")
        return None
    
    # Try to find frame2.png in test data or examples
    img_path = os.path.join(data_dir, "img", target_frame)
    if not os.path.exists(img_path):
        # Try in microflow examples
        img_path = "../microflow_test/examples/frame2.png"
        if not os.path.exists(img_path):
            print(f"Frame2.png not found in test data or microflow examples")
            return None

    # Use the same logic as in process_frame to generate the TFLite prediction
    mask_path = img_path.replace("img", "mask")
    dataset = LoadDataset([img_path], [mask_path])
    image, target = dataset[0]
    image = image.unsqueeze(0)  # Add batch dimension
    target = target.unsqueeze(0)  # Add batch dimension

    print(f"Running inference with {tflite_model_info['name']} on {target_frame}...")
    tflite_pred = get_single_prediction(tflite_model_info, image, target, device)
    tflite_pred_binary = (tflite_pred > 0.5).astype(np.float32)
    if len(tflite_pred_binary.shape) == 4:
        tflite_pred_binary = tflite_pred_binary[0, 0]  # Remove batch and channel dims
    elif len(tflite_pred_binary.shape) == 3:
        tflite_pred_binary = tflite_pred_binary[0]  # Remove batch dim
    # Resize prediction to match original image size (done later if needed)
    
    # Load original image for display
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Could not load image from {img_path}")
        return None
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load microflow prediction
    microflow_pred = cv2.imread(microflow_prediction_path, cv2.IMREAD_GRAYSCALE)
    if microflow_pred is None:
        print(f"Could not load microflow prediction from {microflow_prediction_path}")
        return None
    
    # Convert microflow prediction to binary (assuming it's a grayscale mask)
    microflow_pred = (microflow_pred > 127).astype(np.float32)
    
    # Resize predictions to match original image size
    target_height, target_width = original_img.shape[:2]
    
    if tflite_pred_binary.shape != (target_height, target_width):
        tflite_pred_binary = cv2.resize(tflite_pred_binary, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    if microflow_pred.shape != (target_height, target_width):
        microflow_pred = cv2.resize(microflow_pred, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    def create_comparison_overlay(pred1, pred2):
        """Create colored overlay comparing two predictions"""
        # Ensure predictions are binary
        pred1 = (pred1 > 0.5).astype(np.uint8)
        pred2 = (pred2 > 0.5).astype(np.uint8)
        
        # Create overlay: blue for TFLite only, orange for microflow only, purple for overlap
        overlay = np.zeros((*pred1.shape, 3), dtype=np.float32)
        overlay[(pred1 == 1) & (pred2 == 0)] = [0, 0, 1]      # Blue: TFLite only
        overlay[(pred1 == 0) & (pred2 == 1)] = [1, 0.5, 0]    # Orange: Microflow only
        overlay[(pred1 == 1) & (pred2 == 1)] = [0.5, 0, 0.5]  # Purple: Both predictions
        
        return overlay
    
    # Create matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TFLite vs Microflow Implementation Comparison - Frame2', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # TFLite prediction
    axes[0, 1].imshow(tflite_pred_binary, cmap='gray')
    axes[0, 1].set_title('TFLite Model Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Microflow prediction
    axes[1, 0].imshow(microflow_pred, cmap='gray')
    axes[1, 0].set_title('Microflow Implementation Prediction', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Comparison overlay
    overlay = create_comparison_overlay(tflite_pred_binary, microflow_pred)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Comparison Overlay\n(Blue=TFLite only, Orange=Microflow only, Purple=Both)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Calculate and display metrics
    intersection = np.sum((tflite_pred_binary > 0.5) & (microflow_pred > 0.5))
    union = np.sum((tflite_pred_binary > 0.5) | (microflow_pred > 0.5))
    iou = intersection / union if union > 0 else 0
    
    # Add metrics text
    metrics_text = f'IoU between predictions: {iou:.4f}'
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, fontweight='bold')
    
    return fig

if __name__ == '__main__':
    os.makedirs("exp", exist_ok=True)
    
    DEVICE = "cpu"
    DATA_DIR = "data/processed_data/test"
    
    # Define frames to process
    frames_to_process = [
        ("frame1_0381.png", "frame1_0381_mask.png"),
        ("frame4_0280.png", "frame4_0280_mask.png")
    ]
    
    # Load models once
    bu_net = BU_Net(n_channels=3)
    bu_net.load_state_dict(torch.load("models/BU_Net.pth", map_location=DEVICE))
    
    nano_u_2l = Nano_U(n_channels=3)
    nano_u_2l.load_state_dict(torch.load("models/Nano_U_2L.pth", map_location=DEVICE))
    
    interpreter = tf.lite.Interpreter(model_path="models/Nano_U.tflite")
    interpreter.allocate_tensors()
    
    # Define models
    models = {
        'BU_Net': {'model': bu_net, 'type': 'pytorch', 'name': 'BU_Net'},
        'Nano_U_f32': {'model': nano_u_2l, 'type': 'pytorch', 'name': 'Nano_U_f32'},
        'Nano_U_int8': {'model': interpreter, 'type': 'tflite', 'name': 'Nano_U_int8'}
    }
    
    # Process each frame
    for target_frame, target_mask in frames_to_process:
        print(f"\n--- Processing {target_frame} ---")
        
        fig = process_frame(target_frame, target_mask, models, DEVICE, DATA_DIR)
        
        if fig is not None:
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure to exp directory
            output_path = os.path.join("exp", f"predictions_{target_frame.replace('.png', '.png')}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            
            plt.show()
            plt.close(fig)  # Close figure to free memory
        else:
            print(f"Failed to process {target_frame}")
    
    # Create benchmark comparison figure
    print(f"\n--- Creating Model Comparison ---")
    
    # Create a simple dataset for benchmarking
    img_files = []
    mask_files = []
    for frame, mask in frames_to_process:
        img_path = os.path.join(DATA_DIR, "img", frame)
        mask_path = os.path.join(DATA_DIR, "mask", mask)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_files.append(img_path)
            mask_files.append(mask_path)
    
    if img_files:
        from torch.utils.data import DataLoader
        dataloader = DataLoader(LoadDataset(img_files, mask_files), batch_size=1)
        
        # Define models with paths for size calculation
        models_with_paths = {
            'BU_Net': {'model': bu_net, 'type': 'pytorch', 'path': 'models/BU_Net.pth', 'name': 'BU_Net'},
            'Nano_U_f32': {'model': nano_u_2l, 'type': 'pytorch', 'path': 'models/Nano_U_2L.pth', 'name': 'Nano_U_f32'},
            'Nano_U_int8': {'model': interpreter, 'type': 'tflite', 'path': 'models/Nano_U.tflite', 'pytorch_params': 0, 'name': 'Nano_U_int8'}
        }
        
        # Collect benchmark data
        benchmark_data = {}
        for name, model_info in models_with_paths.items():
            print(f"Benchmarking {model_info['name']}...")
            
            # Measure inference time
            times = measure_inference_time(model_info, dataloader, DEVICE)
            avg_time_ms = np.mean(times) * 1000
            
            # Get model size
            size_kb = get_model_size(model_info['path'])
            
            benchmark_data[model_info['name']] = {
                'size_kb': size_kb,
                'time_ms': avg_time_ms
            }
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(benchmark_data.keys())
        sizes = [benchmark_data[name]['size_kb'] for name in model_names]
        times = [benchmark_data[name]['time_ms'] for name in model_names]
        
        # Define colors for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # Model size comparison
        bars1 = ax1.bar(model_names, sizes, color=colors)
        ax1.set_ylabel('Model Size (KB)')
        ax1.set_title('Model Size Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars1, sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{size:.1f} KB', ha='center', va='bottom', fontweight='bold')
        
        # Inference time comparison
        bars2 = ax2.bar(model_names, times, color=colors)
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time:.2f} ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save comparison figure
        comparison_output_path = os.path.join("exp", "model_comparison.png")
        plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {comparison_output_path}")
        
        plt.show()
        plt.close(fig)
        
        # Print summary table
        print("\n--- Model Performance Summary ---")
        print(f"{'Model':<15} | {'Size (KB)':<10} | {'Time (ms)':<10}")
        print("-" * 40)
        for name in model_names:
            print(f"{name:<15} | {benchmark_data[name]['size_kb']:<10.1f} | {benchmark_data[name]['time_ms']:<10.2f}")
    
    # Create TFLite vs Microflow comparison
    print(f"\n--- Creating TFLite vs Microflow Comparison ---")
    tflite_model_info = models_with_paths['Nano_U_int8']
    comparison_fig = create_tflite_microflow_comparison(tflite_model_info, DEVICE, DATA_DIR)
    
    if comparison_fig is not None:
        plt.tight_layout()
        
        # Save comparison figure
        comparison_output_path = os.path.join("exp", "tflite_vs_microflow_comparison.png")
        comparison_fig.savefig(comparison_output_path, dpi=300, bbox_inches='tight')
        print(f"TFLite vs Microflow comparison saved to {comparison_output_path}")
        
        plt.show()
        plt.close(comparison_fig)
    else:
        print("Failed to create TFLite vs Microflow comparison")
    
    print("\nAll visualizations complete!")