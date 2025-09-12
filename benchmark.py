import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
from torch.utils.data import DataLoader
import tensorflow as tf

from models.BU_Net.BU_Net_model import BU_Net
from models.Nano_U.Nano_U_model import Nano_U
from utils.LoadDataset import LoadDataset

def clear_memory():
    """Clear memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model_parameters(model_info):
    """Calculate the number of trainable parameters in a model.
    
    Args:
        model_info: Dict containing model info with 'model' and 'type' keys
    
    Returns:
        Number of parameters
    """
    if model_info['type'] == 'pytorch':
        return sum(p.numel() for p in model_info['model'].parameters() if p.requires_grad)
    elif model_info['type'] == 'tflite':
        # For TFLite, we need to estimate from the PyTorch equivalent if available
        # or return 0 if we can't determine
        return model_info.get('pytorch_params', 0)
    return 0

def get_model_size(model_path):
    """Get the file size of a model in kilobytes (KB)."""
    return os.path.getsize(model_path) / 1024

def measure_inference_time(model_info, dataloader, device, num_warmup=5, num_iterations=None):
    """Measure inference time for a model.
    
    Args:
        model_info: Dict with 'model', 'type', and other model information
        dataloader: PyTorch DataLoader
        device: Device to run inference on
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to measure (None = all data)
    
    Returns:
        List of inference times in seconds
    """
    times = []
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device)
        model.eval()
    
    print(f"Running {num_warmup} warm-up iterations...")
    clear_memory()
    
    # Warm-up runs
    for _ in range(num_warmup):
        image, _ = next(iter(dataloader))
        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(image.to(device))
        elif model_type == 'tflite':
            _run_tflite_inference(model, image)
        clear_memory()
    
    print("Measuring inference times...")
    iteration_count = 0
    
    with torch.no_grad():
        for image, _ in dataloader:
            if num_iterations and iteration_count >= num_iterations:
                break
                
            image = image.to(device)
            
            start_time = time.time()
            if model_type == 'pytorch':
                _ = model(image)
            elif model_type == 'tflite':
                _run_tflite_inference(model, image)
            end_time = time.time()
            
            times.append(end_time - start_time)
            iteration_count += 1
    
    return times


def measure_accuracy(model_info, dataloader, device, num_iterations=None):
    """Measure model accuracy using IoU metric and cross entropy accuracy.
    
    Args:
        model_info: Dict with 'model', 'type', and other model information
        dataloader: PyTorch DataLoader
        device: Device to run inference on
        num_iterations: Number of iterations to measure (None = all data)
    
    Returns:
        Tuple of (List of IoU scores, List of cross entropy accuracies)
    """
    ious = []
    ce_accuracies = []
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device)
        model.eval()
    
    iteration_count = 0
    
    with torch.no_grad():
        for image, target in dataloader:
            if num_iterations and iteration_count >= num_iterations:
                break
                
            image = image.to(device)
            target = target.to(device)
            
            if model_type == 'pytorch':
                output = model(image)
            elif model_type == 'tflite':
                output = _run_tflite_inference_with_target(model, image, target)
            
            # Calculate IoU
            current_iou = calculate_iou(output, target)
            ious.append(current_iou)
            
            # Calculate cross entropy accuracy
            current_ce_acc = calculate_cross_entropy_accuracy(output, target)
            ce_accuracies.append(current_ce_acc)
            
            iteration_count += 1
    
    return ious, ce_accuracies

def collect_examples(model_info, dataloader, device, num_examples=3):
    """Collect example predictions for visualization.
    
    Args:
        model_info: Dict with 'model', 'type', and other model information
        dataloader: PyTorch DataLoader
        device: Device to run inference on
        num_examples: Number of examples to collect
    
    Returns:
        List of example dictionaries
    """
    examples = []
    model = model_info['model']
    model_type = model_info['type']
    
    if model_type == 'pytorch':
        model.to(device)
        model.eval()
    
    example_count = 0
    
    with torch.no_grad():
        for image, target in dataloader:
            if example_count >= num_examples:
                break
                
            image = image.to(device)
            target = target.to(device)
            
            if model_type == 'pytorch':
                output = model(image)
            elif model_type == 'tflite':
                output = _run_tflite_inference_with_target(model, image, target)
            
            # Calculate IoU for this example
            current_iou = calculate_iou(output, target)
            
            examples.append({
                'image': image.clone().detach().cpu(),
                'target': target.clone().detach().cpu(),
                'output': output.clone().detach().cpu(),
                'iou': current_iou
            })
            example_count += 1
    
    return examples

def _run_tflite_inference(interpreter, image):
    """Helper function to run TFLite inference."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    
    # Process input data
    img_np = image.cpu().numpy()
    input_data = np.zeros(expected_shape, dtype=np.float32)
    
    # Convert from NCHW to NHWC and resize
    for b in range(min(img_np.shape[0], expected_shape[0])):
        for c in range(min(img_np.shape[1], expected_shape[3])):
            original = img_np[b, c]
            resized = cv2.resize(original, (expected_shape[2], expected_shape[1]),
                               interpolation=cv2.INTER_AREA)
            input_data[b, :, :, c] = resized
    
    # Handle quantization
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = np.round(input_data / input_scale + input_zero_point).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

def _run_tflite_inference_with_target(interpreter, image, target):
    """Helper function to run TFLite inference and process output to match target format."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    
    # Process input data
    img_np = image.cpu().numpy()
    input_data = np.zeros(expected_shape, dtype=np.float32)
    
    # Convert from NCHW to NHWC and resize
    for b in range(min(img_np.shape[0], expected_shape[0])):
        for c in range(min(img_np.shape[1], expected_shape[3])):
            original = img_np[b, c]
            resized = cv2.resize(original, (expected_shape[2], expected_shape[1]),
                               interpolation=cv2.INTER_AREA)
            input_data[b, :, :, c] = resized
    
    # Handle quantization
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = np.round(input_data / input_scale + input_zero_point).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle dequantization
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    # Process output to match target format
    output_processed = np.zeros((target.shape[0], target.shape[1], target.shape[2], target.shape[3]), 
                               dtype=np.float32)
    
    if len(output_data.shape) == 4:  # NHWC format
        for b in range(min(output_data.shape[0], target.shape[0])):
            for c in range(min(output_data.shape[3], target.shape[1])):
                output_slice = output_data[b, :, :, c]
                target_h, target_w = target.shape[2], target.shape[3]
                resized_slice = cv2.resize(output_slice, (target_w, target_h),
                                         interpolation=cv2.INTER_LINEAR)
                output_processed[b, c] = resized_slice
    elif len(output_data.shape) == 3:  # NHW format
        for b in range(min(output_data.shape[0], target.shape[0])):
            output_slice = output_data[b]
            target_h, target_w = target.shape[2], target.shape[3]
            resized_slice = cv2.resize(output_slice, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)
            output_processed[b, 0] = resized_slice
    
    return torch.from_numpy(output_processed).to(target.device)

def calculate_iou(pred, target, threshold=0.5):
    """Calculate the Intersection over Union (IoU) for segmentation evaluation."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_cross_entropy_accuracy(pred, target):
    """Calculate accuracy using cross entropy loss for segmentation.
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth binary mask
    
    Returns:
        Accuracy value (1 - normalized_cross_entropy)
    """
    # Apply sigmoid to get probabilities if pred contains logits
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred_probs = torch.sigmoid(pred)
    else:
        pred_probs = pred
    
    # Ensure target is binary (0 or 1)
    target_binary = (target > 0.5).float()
    
    # Calculate cross entropy loss manually to avoid numerical issues
    # CE = -[y*log(p) + (1-y)*log(1-p)]
    eps = 1e-7  # Small epsilon to avoid log(0)
    pred_probs = torch.clamp(pred_probs, eps, 1 - eps)
    
    ce_loss = -(target_binary * torch.log(pred_probs) + 
                (1 - target_binary) * torch.log(1 - pred_probs))
    
    # Calculate mean cross entropy loss
    mean_ce_loss = ce_loss.mean().item()
    
    # Convert to accuracy: higher accuracy = lower loss
    # Normalize by maximum possible loss (ln(2) ≈ 0.693 for binary classification)
    max_loss = np.log(2)
    accuracy = max(0, 1 - (mean_ce_loss / max_loss))
    
    return accuracy

def benchmark_model(model_info, dataloader, device, store_examples=False, num_examples=3):
    """Comprehensive benchmark of a model using separate measurement functions.
    
    Args:
        model_info: Dict containing:
            - 'model': The model object (PyTorch model or TFLite interpreter)
            - 'type': 'pytorch' or 'tflite'
            - 'path': Path to the model file
            - 'pytorch_params': (optional) Number of PyTorch parameters for TFLite models
        dataloader: PyTorch DataLoader
        device: Device to run inference on
        store_examples: Whether to collect example predictions
        num_examples: Number of examples to collect
    
    Returns:
        Dictionary containing all benchmark results
    """
    print(f"Benchmarking {model_info.get('name', 'Unknown')} model...")
    
    # Measure inference time
    print("  - Measuring inference time...")
    times = measure_inference_time(model_info, dataloader, device)
    avg_time_ms = np.mean(times) * 1000
    
    # Measure accuracy
    print("  - Measuring accuracy...")
    ious, ce_accuracies = measure_accuracy(model_info, dataloader, device)
    avg_iou = np.mean(ious)
    avg_ce_accuracy = np.mean(ce_accuracies)
    
    # Collect examples if requested
    examples = []
    if store_examples:
        print(f"  - Collecting {num_examples} examples...")
        examples = collect_examples(model_info, dataloader, device, num_examples)
    
    # Calculate derived metrics
    model_size_kb = get_model_size(model_info['path'])
    
    return {
        "avg_time_ms": avg_time_ms,
        "avg_iou": avg_iou,
        "avg_ce_accuracy": avg_ce_accuracy,
        "examples": examples,
        "size_kb": model_size_kb,
        "params_m": get_model_parameters(model_info) / 1e6
    }

def plot_results(results):
    """Generate bar charts to compare model benchmark results."""
    model_names = list(results.keys())
    
    # Create performance metrics charts
    performance_metrics = {
        'Cross Entropy Accuracy': [v.get('avg_ce_accuracy', 0) for v in results.values()],
        'Model Size (KB)': [v['size_kb'] for v in results.values()],
        'Inference Time (ms)': [v['avg_time_ms'] for v in results.values()],
    }
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('U-Net Models Performance Comparison', fontsize=16)
    axes1 = axes1.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (title, values) in enumerate(performance_metrics.items()):
        bars = axes1[i].bar(model_names, values, color=colors)
        axes1[i].set_title(title)
        axes1[i].set_ylabel(title.split(' ')[-1].strip('()'))
        axes1[i].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            yval = bar.get_height()
            axes1[i].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("exp/benchmark_performance.png")
    print("\nPerformance benchmark charts saved to 'exp/benchmark_performance.png'")
    
    # IoU comparison chart
    fig2, axes2 = plt.subplots(figsize=(10, 6))
    iou_values = [v['avg_iou'] for v in results.values()]
    
    bars = axes2.bar(model_names, iou_values, color=colors)
    axes2.set_title('Segmentation Accuracy (IoU)', fontsize=14)
    axes2.set_ylabel('IoU')
    axes2.set_ylim([0, max(iou_values) * 1.2])  # Add some space above the highest bar
    
    for bar in bars:
        yval = bar.get_height()
        axes2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center',
                 bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.savefig("exp/benchmark_iou.png")
    print("IoU benchmark chart saved to 'exp/benchmark_iou.png'")
    
    # Example predictions visualization
    has_examples = any(len(v.get('examples', [])) > 0 for v in results.values())
    
    if has_examples:
        num_models = len(model_names)
        num_examples = min(len(v.get('examples', [])) for v in results.values() if 'examples' in v)
        
        if num_examples > 0:
            fig3 = plt.figure(figsize=(15, 5 * num_examples))
            gs = fig3.add_gridspec(num_examples, num_models)
            
            for i, model_name in enumerate(model_names):
                model_examples = results[model_name].get('examples', [])
                
                for j in range(min(num_examples, len(model_examples))):
                    example = model_examples[j]
                    ax = fig3.add_subplot(gs[j, i])
                    
                    # Prepare image
                    img = example['image'][0].permute(1, 2, 0).numpy()
                    img = (img * 0.5 + 0.5).clip(0, 1)
                    
                    # Prepare ground truth mask
                    gt_mask = (example['target'][0] > 0.5).float().numpy()
                    
                    # Prepare prediction mask based on model type
                    if model_name == 'Nano_U_int8':
                        pred_probs = torch.sigmoid(example['output'][0])
                        pred_np = pred_probs.numpy()
                        
                        # Take first channel if needed
                        if len(pred_np.shape) == 3:
                            pred_np = pred_np[0]
                        
                        # Ensure we have a 2D mask
                        pred_np = np.squeeze(pred_np)
                        pred_mask = (pred_np > 0.5).astype(np.float32)
                    else:
                        pred_mask = (torch.sigmoid(example['output'][0]) > 0.5).float().numpy()
                    
                    # Create overlay
                    h, w, _ = img.shape
                    overlay = np.zeros((h, w, 3), dtype=np.float32)
                    
                    # Ensure masks have correct dimensions
                    if len(pred_mask.shape) > 2:
                        pred_mask = pred_mask[0]
                    
                    if pred_mask.shape != (h, w):
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    gt_mask_display = gt_mask[0].squeeze()
                    if gt_mask_display.shape != (h, w):
                        gt_mask_display = cv2.resize(gt_mask_display, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Create overlay colors
                    overlay[(gt_mask_display > 0.5) & (pred_mask > 0.5)] = [0, 1, 0]  # True positive (green)
                    overlay[(gt_mask_display > 0.5) & (pred_mask <= 0.5)] = [1, 1, 0]  # False negative (yellow)
                    overlay[(gt_mask_display <= 0.5) & (pred_mask > 0.5)] = [1, 0, 0]  # False positive (red)
                    
                    # Display results
                    ax.imshow(img)
                    ax.imshow(overlay, alpha=0.5)
                    ax.set_title(f"{model_name} - IoU: {example['iou']:.4f}")
                    ax.axis('off')
                    
                    # Add legend to first row
                    if j == 0:
                        green_patch = plt.matplotlib.patches.Patch(color='green', label='True Positive')
                        yellow_patch = plt.matplotlib.patches.Patch(color='yellow', label='False Negative')
                        red_patch = plt.matplotlib.patches.Patch(color='red', label='False Positive')
                        ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='upper right', fontsize=8)
            
            plt.suptitle('Model Prediction Examples', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig("exp/benchmark_examples.png")
            print("Example predictions saved to 'exp/benchmark_examples.png'")
    
    plt.show()

if __name__ == '__main__':
    # Create exp directory if it doesn't exist
    os.makedirs("exp", exist_ok=True)

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 1
    DEVICE = "cpu"
    MODELS_DIR = "models"
    DATA_DIR = "data/processed_data/test"
    NUM_EXAMPLES = 3  # Number of example predictions to show for each model
    
    print(f"Loading dataset from: {DATA_DIR}")
    
    # Load dataset
    img_dir = os.path.join(DATA_DIR, "img")
    mask_dir = os.path.join(DATA_DIR, "mask")
    
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    dataset = LoadDataset(img_files, mask_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Check image dimensions
    sample_img, sample_mask = next(iter(dataloader))
    print(f"Sample image shape from dataloader: {sample_img.shape}")
    print(f"Sample mask shape from dataloader: {sample_mask.shape}")
    
    # Define model paths
    bu_net_path = os.path.join(MODELS_DIR, "BU_Net.pth")
    nano_u_2l_path = os.path.join(MODELS_DIR, "Nano_U_2L.pth")
    provissima2_tflite_path = os.path.join(MODELS_DIR, "provissima2.tflite")

    # Load models
    print("Loading PyTorch models...")
    bu_net = BU_Net(n_channels=3)
    bu_net.load_state_dict(torch.load(bu_net_path, map_location=DEVICE))

    nano_u_2l = Nano_U(n_channels=3) # Assuming Nano_U can be instantiated like this for Nano_U_2L
    nano_u_2l.load_state_dict(torch.load(nano_u_2l_path, map_location=DEVICE))

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=provissima2_tflite_path)
    interpreter.allocate_tensors()

    results = {}

    # Prepare model information dictionaries
    bu_net_info = {
        'model': bu_net,
        'type': 'pytorch',
        'path': bu_net_path,
        'name': 'BU_Net'
    }
    
    nano_u_2l_info = {
        'model': nano_u_2l,
        'type': 'pytorch', 
        'path': nano_u_2l_path,
        'name': 'Nano_U_2L'
    }
    
    provissima2_tflite_info = {
        'model': interpreter,
        'type': 'tflite',
        'path': provissima2_tflite_path,
        'pytorch_params': 0, # Unknown
        'name': 'provissima2_tflite'
    }

    # Benchmark BU_Net
    print("\n--- Starting Benchmark: BU_Net (PyTorch) ---")
    results['BU_Net'] = benchmark_model(bu_net_info, dataloader, DEVICE, 
                                       store_examples=True, num_examples=NUM_EXAMPLES)
    
    # Benchmark Nano_U_2L
    print("\n--- Starting Benchmark: Nano_U_2L (PyTorch) ---")
    results['Nano_U_2L'] = benchmark_model(nano_u_2l_info, dataloader, DEVICE,
                                       store_examples=True, num_examples=NUM_EXAMPLES)

    # Benchmark provissima2 TFLite
    print("\n--- Starting Benchmark: provissima2_tflite (TFLite) ---")
    results['provissima2_tflite'] = benchmark_model(provissima2_tflite_info, dataloader, DEVICE,
                                            store_examples=True, num_examples=NUM_EXAMPLES)
    
    # Display results
    print("\n" + "="*90)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*90)
    print(f"{'Model':<20} | {'Parameters (M)':<15} | {'Size (KB)':<10} | {'IoU':<8} | {'CE Accuracy':<12} | {'Time (ms)':<12}")
    print("-" * 90)
    for name, res in results.items():
        print(f"{name:<20} | {res['params_m']:<15.2f} | {res['size_kb']:<10.2f} | {res['avg_iou']:<8.3f} | {res['avg_ce_accuracy']:<12.3f} | {res['avg_time_ms']:<12.2f}")
    print("-" * 90)

    # Analyze best/worst models
    print("\nGenerating detailed IoU analysis and example predictions...")
    if results:
        models_by_iou = sorted(results.keys(), key=lambda k: results[k]['avg_iou'], reverse=True)
        best_model = models_by_iou[0]
        worst_model = models_by_iou[-1]
        
        print(f"Best IoU: {best_model} with {results[best_model]['avg_iou']:.4f}")
        print(f"Worst IoU: {worst_model} with {results[worst_model]['avg_iou']:.4f}")
        print(f"IoU difference: {results[best_model]['avg_iou'] - results[worst_model]['avg_iou']:.4f}")

    # Generate charts
    plot_results(results)