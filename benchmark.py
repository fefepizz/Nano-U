import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from glob import glob
from torchvision import transforms

from models.BU_Net.BU_Net_model import BU_Net
from models.Nano_U.Nano_U_model import Nano_U
from utils.LoadDataset import LoadDataset

def get_model_parameters(model):
    """Calculate the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model_path):
    """Get the file size of a model in megabytes (MB)."""
    return os.path.getsize(model_path) / (1024 * 1024)

def calculate_iou(pred, target, threshold=0.5):
    """Calculate the Intersection over Union (IoU) for segmentation evaluation."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def benchmark_inference(model, dataloader, device, model_type='pytorch'):
    """Perform inference on a dataset and measure performance metrics."""
    times, cpus, rams_used, ious = [], [], [], []
    
    if model_type == 'pytorch':
        model.to(device)
        model.eval()
    
    process = psutil.Process(os.getpid())
    print("Running warm-up...")
    for _ in range(5):
        if dataloader:
            image, _ = next(iter(dataloader))
            if model_type == 'pytorch':
                model(image.to(device))

    print("Starting measurements...")

    with torch.no_grad():
        for image, target in dataloader:
            image = image.to(device)
            target = target.to(device)

            psutil.cpu_percent(interval=0.1)
            ram_before_inference = process.memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            if model_type == 'pytorch':
                output = model(image)
            elif model_type == 'tflite':
                interpreter = model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                input_data = image.cpu().numpy()
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = np.round(input_data / input_scale + input_zero_point).astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                if output_details[0]['dtype'] == np.uint8:
                     output_scale, output_zero_point = output_details[0]['quantization']
                     output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
                output = torch.from_numpy(output_data).to(device)

            end_time = time.time()
            ram_after_inference = process.memory_info().rss / (1024 * 1024)
            cpu_usage = psutil.cpu_percent(interval=0.1)

            times.append(end_time - start_time)
            cpus.append(cpu_usage)
            rams_used.append(ram_after_inference - ram_before_inference)
            ious.append(calculate_iou(output, target))
            
    return {
        "avg_time_ms": np.mean(times) * 1000,
        "avg_cpu_percent": np.mean(cpus),
        "avg_ram_mb": np.mean(rams_used),
        "avg_iou": np.mean(ious)
    }

def plot_results(results):
    """Generate bar charts to compare model benchmark results."""
    model_names = list(results.keys())
    
    metrics = {
        'Parameters (Millions)': [v.get('params_m', 0) for v in results.values()],
        'Model Size (MB)': [v['size_mb'] for v in results.values()],
        'Accuracy (IoU)': [v['avg_iou'] for v in results.values()],
        'Inference Time (ms)': [v['avg_time_ms'] for v in results.values()],
        'CPU Usage (%)': [v['avg_cpu_percent'] for v in results.values()],
        'RAM Usage (MB)': [v['avg_ram_mb'] for v in results.values()],
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('U-Net Models Benchmark Comparison', fontsize=16)
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (title, values) in enumerate(metrics.items()):
        bars = axes[i].bar(model_names, values, color=colors)
        axes[i].set_title(title)
        axes[i].set_ylabel(title.split(' ')[-1].strip('()'))
        axes[i].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center',
                         bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_results.png")
    print("\nBenchmark charts saved to 'benchmark_results.png'")
    plt.show()

if __name__ == '__main__':
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 1
    DEVICE = "cpu"
    MODELS_DIR = "models"
    DATA_DIR = "data/processed_data/10"
    
    print(f"Loading dataset from: {DATA_DIR}")
    
    img_dir = os.path.join(DATA_DIR, "img")
    mask_dir = os.path.join(DATA_DIR, "mask")
    
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    dataset = LoadDataset(img_files, mask_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    bu_net_path = os.path.join(MODELS_DIR, "bu_net.pth")
    nano_u_path = os.path.join(MODELS_DIR, "nano_u.pth")
    nano_u_tflite_path = os.path.join(MODELS_DIR, "nano_u_int8.tflite")

    print("Loading PyTorch models...")
    bu_net = BU_Net()
    bu_net.load_state_dict(torch.load(bu_net_path, map_location=DEVICE))

    nano_u = Nano_U()
    nano_u.load_state_dict(torch.load(nano_u_path, map_location=DEVICE))

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=nano_u_tflite_path)
    interpreter.allocate_tensors()

    results = {}

    print("\n--- Starting Benchmark: BU_Net (PyTorch) ---")
    benchmark_data_bu = benchmark_inference(bu_net, dataloader, DEVICE, model_type='pytorch')
    results['BU_Net'] = {
        'params_m': get_model_parameters(bu_net) / 1e6,
        'size_mb': get_model_size(bu_net_path),
        **benchmark_data_bu
    }
    
    print("\n--- Starting Benchmark: Nano_U (PyTorch) ---")
    benchmark_data_nano = benchmark_inference(nano_u, dataloader, DEVICE, model_type='pytorch')
    results['Nano_U'] = {
        'params_m': get_model_parameters(nano_u) / 1e6,
        'size_mb': get_model_size(nano_u_path),
        **benchmark_data_nano
    }

    print("\n--- Starting Benchmark: Nano_U_int8 (TFLite) ---")
    benchmark_data_tflite = benchmark_inference(interpreter, dataloader, DEVICE, model_type='tflite')
    results['Nano_U_int8'] = {
        'params_m': results['Nano_U']['params_m'],
        'size_mb': get_model_size(nano_u_tflite_path),
        **benchmark_data_tflite
    }
    
    print("\n" + "="*95)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*95)
    print(f"{'Model':<15} | {'Parameters (M)':<15} | {'Size (MB)':<10} | {'IoU':<8} | {'Time (ms)':<12} | {'CPU (%)':<10} | {'RAM (MB)':<10}")
    print("-" * 95)
    for name, res in results.items():
        print(f"{name:<15} | {res['params_m']:<15.2f} | {res['size_mb']:<10.2f} | {res['avg_iou']:<8.3f} | {res['avg_time_ms']:<12.2f} | {res['avg_cpu_percent']:<10.2f} | {res['avg_ram_mb']:<10.2f}")
    print("-" * 95)

    plot_results(results)