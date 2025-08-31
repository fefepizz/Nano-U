import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import cv2
from torch.utils.data import DataLoader
import tensorflow as tf

from models.BU_Net.BU_Net_model import BU_Net
from models.Nano_U.Nano_U_model import Nano_U
from utils.LoadDataset import LoadDataset

def get_model_parameters(model):
    """Calculate the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model_path):
    """Get the file size of a model in megabytes (MB)."""
    return os.path.getsize(model_path) / (1024 * 1024)
    
def calculate_tops(model, input_shape, time_ms):
    """Calculate Trillion Operations Per Second (TOPS) for a PyTorch model.
    
    Args:
        model: PyTorch model
        input_shape: Tuple representing input shape (batch_size, channels, height, width)
        time_ms: Inference time in milliseconds
    
    Returns:
        TOPS value (float)
    """
    # Calculate MACs (Multiply-Accumulate operations)
    # This is a simple estimation based on model parameters and input dimensions
    params = get_model_parameters(model)
    input_size = np.prod(input_shape)
    
    # Estimate operations: each parameter typically requires at least 2 operations (multiply and add)
    # and is used at least once per input element (very rough approximation)
    estimated_ops = params * 2 * input_size
    
    # Convert to TOPS (Trillion Operations Per Second)
    time_s = time_ms / 1000.0  # Convert ms to seconds
    tops = estimated_ops / (time_s * 1e12)  # Trillion ops per second
    
    return tops

def calculate_iou(pred, target, threshold=0.5):
    """Calculate the Intersection over Union (IoU) for segmentation evaluation."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def plot_prediction(image, actual_mask, predicted_mask, title):
    """
    Plot comparison between input image, ground truth mask, and model prediction.
    
    Args:
        image (torch.Tensor): Input image tensor
        actual_mask (torch.Tensor): Ground truth mask tensor
        predicted_mask (torch.Tensor): Predicted mask tensor
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The generated comparison figure
    """
    # Prepare image
    img = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    # Prepare masks
    gt_mask = actual_mask.cpu().squeeze().numpy()
    pred_mask = predicted_mask.cpu().squeeze().numpy()

    # Ensure masks are binary
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Create a white background and set yellow where mask==1
    gt_mask_rgb = np.ones((*gt_mask.shape, 3), dtype=np.float32)  # white background
    gt_mask_rgb[gt_mask == 1] = [1, 1, 0]  # yellow for foreground

    # Overlay: yellow for GT only, red for wrong prediction only, green for overlap
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    overlay[(gt_mask == 1) & (pred_mask == 0)] = [1, 1, 0]  # Yellow for ground truth only
    overlay[(gt_mask == 0) & (pred_mask == 1)] = [1, 0, 0]  # Red for false positive
    overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 1, 0]  # Green for true positive

    # Calculate IoU for this prediction
    intersection = np.sum((gt_mask == 1) & (pred_mask == 1))
    union = np.sum((gt_mask == 1) | (pred_mask == 1))
    iou = intersection / (union + 1e-6)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{title} - IoU: {iou:.4f}", fontsize=14)
    
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_mask_rgb)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(img, alpha=0.7)
    axs[2].imshow(overlay, alpha=0.5)
    axs[2].set_title("Prediction Overlay")
    axs[2].axis('off')

    # Legend
    yellow_patch = plt.matplotlib.patches.Patch(color='yellow', label='Ground Truth Only')
    red_patch = plt.matplotlib.patches.Patch(color='red', label='False Positive')
    green_patch = plt.matplotlib.patches.Patch(color='green', label='True Positive')
    axs[2].legend(handles=[yellow_patch, red_patch, green_patch], loc='lower right', fontsize=8)

    plt.tight_layout()
    return fig

def benchmark_inference(model, dataloader, device, model_type='pytorch', verbose=False, store_examples=False, num_examples=3):
    """Perform inference on a dataset and measure performance metrics.
    
    Uses a threading approach to continuously monitor memory usage during inference,
    capturing the peak memory allocation during model execution.
    
    Args:
        model: PyTorch model or TFLite interpreter
        dataloader: PyTorch dataloader
        device: Device to run inference on ('cpu' or 'cuda')
        model_type: Type of model ('pytorch' or 'tflite')
        verbose: Print verbose information
        store_examples: Whether to store example predictions for visualization
        num_examples: Number of examples to store if store_examples is True
    
    Returns:
        Dictionary containing benchmark results:
        - avg_time_ms: Average inference time in milliseconds
        - avg_cpu_percent: Average CPU usage during inference
        - avg_ram_mb: Average RAM usage during inference in KB (not MB despite the name)
        - avg_iou: Average IoU score
        - examples: List of example predictions (if store_examples is True)
    """
    times, cpus, rams_used, ious = [], [], [], []
    examples = []
    
    if model_type == 'pytorch':
        model.to(device)
        model.eval()
    
    process = psutil.Process(os.getpid())
    print("Running warm-up...")
    
    # Clear any existing cached memory
    import gc
    gc.collect()
    
    # Perform warm-up runs
    for _ in range(5):
        if dataloader:
            image, _ = next(iter(dataloader))
            if model_type == 'pytorch':
                out = model(image.to(device))
                del out
                gc.collect()
            elif model_type == 'tflite':
                interpreter = model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                input_data = image.cpu().numpy()
                input_data = np.transpose(input_data, (0, 2, 3, 1))  # NCHW -> NHWC
                
                # Resize to match expected dimensions
                expected_shape = input_details[0]['shape']
                if input_data.shape != tuple(expected_shape):
                    resized_input = np.zeros(expected_shape, dtype=input_data.dtype)
                    min_batch = min(input_data.shape[0], expected_shape[0])
                    min_height = min(input_data.shape[1], expected_shape[1])
                    min_width = min(input_data.shape[2], expected_shape[2])
                    min_channels = min(input_data.shape[3], expected_shape[3])
                    resized_input[:min_batch, :min_height, :min_width, :min_channels] = \
                        input_data[:min_batch, :min_height, :min_width, :min_channels]
                    input_data = resized_input
                
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = np.round(input_data / input_scale + input_zero_point).astype(np.uint8)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

    print("Starting measurements...")
    example_count = 0

    with torch.no_grad():
        for image, target in dataloader:
            image = image.to(device)
            target = target.to(device)

            # Reset measurement variables
            memory_samples = []
            cpu_samples = []
            
            # Set up resource monitoring
            import threading
            monitoring_active = True
            
            def monitor_resources():
                while monitoring_active:
                    memory_samples.append(process.memory_info().rss / 1024)  # KB
                    cpu_samples.append(psutil.cpu_percent(interval=0.01))
                    time.sleep(0.01)  # Sample every 10ms
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Perform inference while monitoring resources
            start_time = time.time()
            
            if model_type == 'pytorch':
                output = model(image)
            elif model_type == 'tflite':
                interpreter = model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                expected_shape = input_details[0]['shape']
                
                # Process input data for TFLite
                img_np = image.cpu().numpy()
                input_data = np.zeros(expected_shape, dtype=np.float32)
                
                # Convert from NCHW to NHWC and resize
                for b in range(min(img_np.shape[0], expected_shape[0])):
                    for c in range(min(img_np.shape[1], expected_shape[3])):
                        original = img_np[b, c]
                        resized = cv2.resize(original, (expected_shape[2], expected_shape[1]),
                                           interpolation=cv2.INTER_AREA)
                        input_data[b, :, :, c] = resized
                
                # Process target data for proper IoU calculation
                target_np = target.cpu().numpy()
                target_resized = np.zeros((target_np.shape[0], target_np.shape[1], expected_shape[1], expected_shape[2]), 
                                        dtype=target_np.dtype)
                
                for b in range(min(target_np.shape[0], target_resized.shape[0])):
                    for c in range(min(target_np.shape[1], target_resized.shape[1])):
                        original_mask = target_np[b, c]
                        target_resized[b, c] = cv2.resize(original_mask, 
                                                       (expected_shape[2], expected_shape[1]),
                                                       interpolation=cv2.INTER_NEAREST)
                
                target = torch.from_numpy(target_resized).to(target.device)
                
                # Handle quantized model
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = np.round(input_data / input_scale + input_zero_point).astype(np.uint8)

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Handle dequantization
                if output_details[0]['dtype'] == np.uint8:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                
                # Process output data to correct format
                output_processed = np.zeros((target.shape[0], target.shape[1], target.shape[2], target.shape[3]), 
                                         dtype=np.float32)
                
                # Handle different output formats
                if len(output_data.shape) == 4:  # NHWC format
                    for b in range(min(output_data.shape[0], target.shape[0])):
                        for c in range(min(output_data.shape[3], target.shape[1])):
                            output_slice = output_data[b, :, :, c]
                            target_h, target_w = target.shape[2], target.shape[3]
                            resized_slice = cv2.resize(output_slice, 
                                                    (target_w, target_h),
                                                    interpolation=cv2.INTER_LINEAR)
                            output_processed[b, c] = resized_slice
                            
                elif len(output_data.shape) == 3:  # NHW format (single channel)
                    for b in range(min(output_data.shape[0], target.shape[0])):
                        output_slice = output_data[b]
                        target_h, target_w = target.shape[2], target.shape[3]
                        resized_slice = cv2.resize(output_slice, 
                                                 (target_w, target_h),
                                                 interpolation=cv2.INTER_LINEAR)
                        output_processed[b, 0] = resized_slice
                
                # Convert back to PyTorch tensor
                output = torch.from_numpy(output_processed).to(device)

            end_time = time.time()
            
            # Stop the monitoring thread
            monitoring_active = False
            monitor_thread.join(timeout=1.0)
            
            # Process resource measurements
            if memory_samples:
                base_memory = memory_samples[0] if memory_samples else 0
                ram_peaks = [max(0, m - base_memory) for m in memory_samples]
                peak_ram_usage = max(ram_peaks) if ram_peaks else 0
                
                # Fallback for small measurements
                if peak_ram_usage < 1024:
                    if model_type == 'pytorch':
                        peak_ram_usage = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024
                    else:
                        model_file_size = os.path.getsize(nano_u_tflite_path) / 1024
                        peak_ram_usage = model_file_size
                
                rams_used.append(peak_ram_usage)
            else:
                # Fallback if monitoring failed
                if model_type == 'pytorch':
                    rams_used.append(sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024)
                else:
                    model_file_size = os.path.getsize(nano_u_tflite_path) / 1024
                    rams_used.append(model_file_size)
            
            # Process CPU measurements
            if cpu_samples:
                avg_cpu = sum(cpu_samples) / len(cpu_samples)
                cpus.append(avg_cpu)
            else:
                cpus.append(psutil.cpu_percent(interval=0.1))
            
            # Convert output logits to probabilities for IoU calculation
            if output.shape[1] == 1:
                pred_mask = torch.sigmoid(output) > 0.5
            else:
                pred_mask = output > 0.5
                
            # Calculate IoU
            current_iou = calculate_iou(output, target)
            
            times.append(end_time - start_time)
            ious.append(current_iou)
            
            # Store example for visualization
            if store_examples and example_count < num_examples:
                examples.append({
                    'image': image.clone().detach().cpu(),
                    'target': target.clone().detach().cpu(),
                    'output': output.clone().detach().cpu(),
                    'iou': current_iou
                })
                example_count += 1
            
    return {
        "avg_time_ms": np.mean(times) * 1000,
        "avg_cpu_percent": np.mean(cpus),
        "avg_ram_mb": np.mean(rams_used),  # RAM usage in KB (variable name kept for backward compatibility)
        "avg_iou": np.mean(ious),
        "examples": examples if store_examples else []
    }

def plot_results(results):
    """Generate bar charts to compare model benchmark results."""
    model_names = list(results.keys())
    
    # Create performance metrics charts
    performance_metrics = {
        'Parameters (Millions)': [v.get('params_m', 0) for v in results.values()],
        'Model Size (MB)': [v['size_mb'] for v in results.values()],
        'Inference Time (ms)': [v['avg_time_ms'] for v in results.values()],
        'CPU Usage (%)': [v['avg_cpu_percent'] for v in results.values()],
        'RAM Usage (KB)': [v['avg_ram_mb'] for v in results.values()],
        'Performance (TOPS)': [v.get('tops', 0) for v in results.values()],
    }
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
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
    plt.savefig("benchmark_performance.png")
    print("\nPerformance benchmark charts saved to 'benchmark_performance.png'")
    
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
    plt.savefig("benchmark_iou.png")
    print("IoU benchmark chart saved to 'benchmark_iou.png'")
    
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
            plt.savefig("benchmark_examples.png")
            print("Example predictions saved to 'benchmark_examples.png'")
    
    plt.show()

if __name__ == '__main__':
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
    nano_u_path = os.path.join(MODELS_DIR, "Nano_U.pth")
    nano_u_tflite_path = os.path.join(MODELS_DIR, "Nano_U_int8.tflite")

    # Load models
    print("Loading PyTorch models...")
    bu_net = BU_Net(n_channels=3)
    bu_net.load_state_dict(torch.load(bu_net_path, map_location=DEVICE))

    nano_u = Nano_U(n_channels=3)
    nano_u.load_state_dict(torch.load(nano_u_path, map_location=DEVICE))

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=nano_u_tflite_path)
    interpreter.allocate_tensors()

    # Input shape for TOPS calculation
    input_shape = (BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1])
    results = {}

    # Benchmark BU_Net
    print("\n--- Starting Benchmark: BU_Net (PyTorch) ---")
    benchmark_data_bu = benchmark_inference(bu_net, dataloader, DEVICE, model_type='pytorch', 
                                          store_examples=True, num_examples=NUM_EXAMPLES)
    bu_net_tops = calculate_tops(bu_net, input_shape, benchmark_data_bu['avg_time_ms'])
    results['BU_Net'] = {
        'params_m': get_model_parameters(bu_net) / 1e6,
        'size_mb': get_model_size(bu_net_path),
        'tops': bu_net_tops,
        **benchmark_data_bu
    }
    
    # Benchmark Nano_U
    print("\n--- Starting Benchmark: Nano_U (PyTorch) ---")
    benchmark_data_nano = benchmark_inference(nano_u, dataloader, DEVICE, model_type='pytorch', 
                                            store_examples=True, num_examples=NUM_EXAMPLES)
    nano_u_tops = calculate_tops(nano_u, input_shape, benchmark_data_nano['avg_time_ms'])
    results['Nano_U'] = {
        'params_m': get_model_parameters(nano_u) / 1e6,
        'size_mb': get_model_size(nano_u_path),
        'tops': nano_u_tops,
        **benchmark_data_nano
    }

    # Benchmark Nano_U TFLite
    print("\n--- Starting Benchmark: Nano_U_int8 (TFLite) ---")
    benchmark_data_tflite = benchmark_inference(interpreter, dataloader, DEVICE, model_type='tflite', 
                                              store_examples=True, num_examples=NUM_EXAMPLES)
    nano_u_int8_tops = calculate_tops(nano_u, input_shape, benchmark_data_tflite['avg_time_ms'])
    results['Nano_U_int8'] = {
        'params_m': results['Nano_U']['params_m'],
        'size_mb': get_model_size(nano_u_tflite_path),
        'tops': nano_u_int8_tops,
        **benchmark_data_tflite
    }
    
    # Display results
    print("\n" + "="*115)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*115)
    print(f"{'Model':<15} | {'Parameters (M)':<15} | {'Size (MB)':<10} | {'IoU':<8} | {'Time (ms)':<12} | {'CPU (%)':<10} | {'RAM (KB)':<10} | {'TOPS':<10}")
    print("-" * 115)
    for name, res in results.items():
        print(f"{name:<15} | {res['params_m']:<15.2f} | {res['size_mb']:<10.2f} | {res['avg_iou']:<8.3f} | {res['avg_time_ms']:<12.2f} | {res['avg_cpu_percent']:<10.2f} | {res['avg_ram_mb']:<10.2f} | {res['tops']:<10.4f}")
    print("-" * 115)

    # Analyze best/worst models
    print("\nGenerating detailed IoU analysis and example predictions...")
    models_by_iou = sorted(results.keys(), key=lambda k: results[k]['avg_iou'], reverse=True)
    best_model = models_by_iou[0]
    worst_model = models_by_iou[-1]
    
    print(f"Best IoU: {best_model} with {results[best_model]['avg_iou']:.4f}")
    print(f"Worst IoU: {worst_model} with {results[worst_model]['avg_iou']:.4f}")
    print(f"IoU difference: {results[best_model]['avg_iou'] - results[worst_model]['avg_iou']:.4f}")

    # Generate charts
    plot_results(results)