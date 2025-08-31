"""
Metrics and Visualization Utilities for Nano-U

This module provides functions for visualizing training metrics and model predictions.
It includes functions for plotting loss and accuracy curves, as well as visualizing
the segmentation masks produced by the model compared to ground truth.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_metrics(train_losses, val_losses, train_accs, val_accs, epochs):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_losses (list): List of training losses for each epoch
        val_losses (list): List of validation losses for each epoch
        train_accs (list): List of training accuracies for each epoch
        val_accs (list): List of validation accuracies for each epoch
        epochs (int): Total number of epochs
        
    Returns:
        matplotlib.figure.Figure: The generated metrics figure
    """
    # Plot training and validation metrics over epochs
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
     
    
# plot the image and the prediction
def plot_prediction(image, actual_mask, predicted_mask):
    """
    Plot input image, ground truth mask, and prediction overlay.
    
    This function visualizes the model's prediction against the ground truth mask:
    - Ground truth only is shown in yellow
    - Prediction only (false positives) is shown in red
    - Overlap (true positives) is shown in green
    
    Args:
        image (torch.Tensor): Input image tensor
        actual_mask (torch.Tensor): Ground truth mask tensor
        predicted_mask (torch.Tensor): Predicted mask tensor
        
    Returns:
        matplotlib.figure.Figure: The generated comparison figure
    """
    # Plots the input image, the ground truth mask, and an overlay of ground truth and predicted masks.
    # Ground truth mask is shown in green, predicted mask in red, and overlap in yellow.
    
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
    overlay[(gt_mask == 1) & (pred_mask == 0)] = [1, 1, 0]
    overlay[(gt_mask == 0) & (pred_mask == 1)] = [1, 0, 0]
    overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 1, 0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_mask_rgb)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(img, alpha=0.7)
    axs[2].imshow(overlay, alpha=0.5)
    axs[2].set_title("Overlay: GT Only (Yellow), Prediction Only (Red), Overlap (Green)")
    axs[2].axis('off')

    # Legend
    yellow_patch = mpatches.Patch(color='yellow', label='GT Only')
    red_patch = mpatches.Patch(color='red', label='Prediction Only')
    green_patch = mpatches.Patch(color='green', label='Overlap')
    axs[2].legend(handles=[yellow_patch, red_patch, green_patch], loc='lower right')

    plt.tight_layout()
    plt.show()
