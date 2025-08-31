"""
Neural Network Implementation Utilities with PyTorch (SAMPLE/TEMPLATE)

This module serves as a sample/template for learning PyTorch and its application in the Nano-U project.
It provides example implementations of neural network components, including:
1. Custom dataset classes for loading image-mask pairs
2. Neural network architecture definition with efficient convolution blocks
3. Training and evaluation functions
4. Visualization utilities for network outputs and training metrics

The network architecture is inspired by MobileNet, EfficientNet, and U-Net designs,
and serves as a learning template for understanding deep learning implementation patterns.
This is primarily for educational purposes to understand PyTorch fundamentals.
"""

# Dependencies

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import cv2


torch.cuda.empty_cache()
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# make sure to set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# define the the subclass of the torch.utils.data.Dataset class to use the dataset in the dataloader
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for general-purpose data loading.
    
    This class provides a flexible dataset implementation that can handle
    both supervised (with labels) and unsupervised (without labels) data.
    
    Args:
        X (list/array): Input data samples
        y (list/array, optional): Target labels if available. Defaults to None.
        transform (callable, optional): Transformations to apply to input data
    """
    
    # When initialized, at least x should be passed to the dataset class
    # y may not be passed if the dataset is not used for supervised learning
    # transform is used to apply transformations to the data like normalization, augmentation, etc.
    
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    # __len__ method returns the number of samples in the dataset
    def __len__(self):
        return len(self.X)
    
    # __getitem__ method retrieves a sample from the dataset at the given index
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # with the transform applied if provided.
        if self.transform is not None:
            x = self.transform(x)
            
        # If y is None, it returns only x
        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    """
    Neural Network architecture for segmentation tasks.
    
    This network implements an efficient encoder-decoder architecture inspired by
    MobileNet, EfficientNet and U-Net. It uses depthwise separable convolutions
    to reduce parameter count while maintaining performance.
    
    Args:
        e_channels (list): List of channel dimensions for encoder blocks
        b_channels (list): List of channel dimensions for bottleneck blocks
        d_channels (list): List of channel dimensions for decoder blocks
        strides (list): List of stride values for all blocks
    """
    
    # the net is inspired by:
    # MobilNet -> Depthwise Separable Convolution, Relu6 (width multiplier)
    # EfficientNet -> Depthwise Separable Convolution, (equation to choose parameters for best efficience), 
    #                 (Expansion after depthwise conv and Swish activation function)
    # U-Net -> Encoder-Decoder architecture, skip (connections)
    
    # (in brackets features are not implemented, also for complexity in MicroFlow implementation)
    # The network needs to be lightweight and efficient.
       
    def __init__(self, e_channels, b_channels, d_channels, strides):
        
        # Call the parent class constructor
        super(NeuralNetwork, self).__init__()
        
        # ReLU6 activation function
        self.relu6 = nn.ReLU6()
  
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.4)
        
        # intial convolutional layer, followed by batch normalization
        out_channels = e_channels[0] 
        # out channels? ###############################################################################
        self.conv_init = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
        self.bn_init = nn.BatchNorm2d(out_channels)
        
        # Use ModuleList to store multiple layers of the same type in a list
        
        # Define depthwise (dw) separable convolution blocks
        self.dw_conv_layers = nn.ModuleList()
        self.dw_bn_layers = nn.ModuleList()
        # Define pointwise (pw) convolution blocks
        self.pw_conv_layers = nn.ModuleList()
        self.pw_bn_layers = nn.ModuleList()
        
        # Note that a depthwise separable convolution consists of a depthwise convolution followed by a pointwise convolution
        # The depthwise convolution applies a single filter to each input channel, while the pointwise convolution combines the outputs of the depthwise convolution.
        
        # network architecture after initial convolution and final fully connected layer
        ###################################################################################################################
        in_channels = e_channels[0]
        encoder_layers = len(e_channels)
        bottleneck_layers = len(b_channels)
        
        # Encoder: Downsampling layers
        for out_channels, stride in zip(e_channels, strides[:encoder_layers]):
            
            # Depthwise convolution (separable by channel), followed by batch normalization
            self.dw_conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels))
            self.dw_bn_layers.append(nn.BatchNorm2d(in_channels))
            
            # Pointwise convolution (1x1 conv to change channels), followed by batch normalization
            self.pw_conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.pw_bn_layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
        
        # Bottleneck layers
        for out_channels, stride in zip(b_channels, strides[encoder_layers:encoder_layers + bottleneck_layers]):
            
            # Depthwise convolution (separable by channel), followed by batch normalization
            self.dw_conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels))
            self.dw_bn_layers.append(nn.BatchNorm2d(in_channels))
            
            # Pointwise convolution (1x1 conv to change channels), followed by batch normalization
            self.pw_conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.pw_bn_layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
        
        # Decoder: Upsampling layers
        for out_channels, stride in zip(d_channels, strides[encoder_layers + bottleneck_layers:]):
            
            # Depthwise convolution (separable by channel), followed by batch normalization
            self.dw_conv_layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1, groups=in_channels))
            self.dw_bn_layers.append(nn.BatchNorm2d(in_channels))
            
            # Pointwise convolution (1x1 conv to change channels), followed by batch normalization
            self.pw_conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.pw_bn_layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
            
        # Max pooling
        # in teoria meglio di avg pooling #####################################################################################################
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final layer that maps to the mask (64x48) dovrebbe essere ok ##############################################################################
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    # forward method defines the flow of data through the network
    # It takes the input x and passes it through the layers defined in __init__
    def forward(self, x):    

        # Initial convolution (layer 1)
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu6(x)
    
        # Iterate through the depthwise and pointwise convolution layers
        for i in range(len(self.dw_conv_layers)):
            
            # Depthwise convolution
            x = self.dw_conv_layers[i](x)
            x = self.dw_bn_layers[i](x)
            x = self.relu6(x)
            
            # Pointwise convolution
            x = self.pw_conv_layers[i](x)
            x = self.pw_bn_layers[i](x)
            x = self.relu6(x)
            
            # Apply dropout for regularization
            x = self.dropout(x)
        
        # Global average pooling and reshape 
        # ok ridimensionamento così? ########################################################################################################
        # torch assumes the size is (height, width)
        x = nn.functional.interpolate(x, size=(48,64), mode="bilinear", align_corners=False)  # Ensure output matches label dimensions
        
        # Map to the correct number of output classes
        x = self.final_conv(x)
        
        return x


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    """
    Train the neural network model.
    
    This function handles the complete training process including:
    - Training and validation loops
    - Metric tracking and reporting
    - Model checkpointing (saving best model)
    - Performance visualization
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs (int): Number of training epochs
        
    Returns:
        nn.Module: The trained model
    """
    
    # Saves the best validation value, when updated the model's state is saved
    best_val_acc = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training phase
    for epoch in range(num_epochs):
        
        # set the model to training mode
        # eg activate dropout layers, batch normalization, etc. (disabled in eval mode)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Wrap the train_loader with tqdm for progress tracking
        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=True, ncols=100)
        
        for inputs, labels in train_loop:
            
            # Move data to the device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients because gradients accumulate by default
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
                        
            # Compute the loss using the criterion (loss function)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            
            # Compute the gradients of the loss with the respect to the model parameters,
            loss.backward()
            
            # update the model parameters using the optimizer
            optimizer.step()
            
            
            # Statistics
    
            # accumulate the loss for the current batch
            running_loss += loss.item() * inputs.size(0)
             
            # Pixel-wise accuracy for binary mask
            preds = (torch.sigmoid(outputs) > 0.5)
            labels_bin = (labels > 0.5)
            correct += (preds == labels_bin).sum().item()
            total += labels.numel()
       
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Update tqdm description with current metrics
        train_loop.set_description(f"Epoch {epoch+1}/{num_epochs} (Train) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        
        # Validation phase
        
        # set the model to evaluation mode
        model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # disable the gradient computation temporarily to save memory and computations (not needed for validation)
        with torch.no_grad():
            
            # Wrap the val_loader with tqdm for progress tracking
            val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)", leave=False)
            
            for inputs, labels in val_loop:
                
                # Move data to the device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass and compute the loss (no backward pass needed)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                              
                # Compute the statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Pixel-wise accuracy for binary mask
                preds = (torch.sigmoid(outputs) > 0.5)
                labels_bin = (labels > 0.5)
                
                val_correct += (preds == labels_bin).sum().item()
                val_total += labels.numel()
            
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Update tqdm description with current metrics
            val_loop.set_description(f"Epoch {epoch+1}/{num_epochs} (Validation) Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, 'f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc            
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, num_epochs)
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model


def plot_metrics(train_losses, val_losses, train_accs, val_accs, epochs):
    
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
    axs[2].set_title("Overlay: GT (Yellow), Wrong pred (Red), Overlap (Green)")
    axs[2].axis('off')

    # Legend
    yellow_patch = mpatches.Patch(color='yellow', label='Correct Label')
    red_patch = mpatches.Patch(color='red', label='Wrong Pixel')
    green_patch = mpatches.Patch(color='green', label='Correct Pixel')
    axs[2].legend(handles=[yellow_patch, red_patch, green_patch], loc='lower right')

    plt.tight_layout()
    plt.show()
    
    

# Main execution
def main():
    """
    Main execution function for training and evaluating the neural network.
    
    This function:
    1. Sets up hyperparameters and network architecture
    2. Loads and prepares the dataset
    3. Initializes the model
    4. Trains the model and evaluates performance
    5. Visualizes sample predictions
    """
    
    # Hyperparameters
    
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 20
    
    ################################################################################################
    
    # Channel configuration for each block
    # Encoder path (increasing channels, decreasing spatial dimensions)
    e_channels = [64, 64, 128, 128, 256, 256, 512, 512]  # Deeper and narrower encoder
    e_strides = [2, 2, 2, 2, 2, 2, 2, 2, 2]            # Strides for each encoder layer
    
    # Bottleneck
    b_channels = [1024, 1024, 1024]           # Deeper and narrower bottleneck
    b_strides = [1, 1, 1]                    # Strides for bottleneck layers
           
    # Decoder path (decreasing channels)
    d_channels = [512, 512, 256, 256, 128, 128, 64, 64]      # Deeper and narrower decoder
    d_strides = [2, 2, 2, 2, 2, 2, 2, 2, 2]           # Strides for each decoder layer
     
    ################################################################################################# 
        
    # Combine encoder-bottleneck-decoder
    strides = e_strides + b_strides + d_strides
    
    # Scelgo trasformazioni ###########################################################################################à
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
    ])
    
    
    # Define paths for images and masks
    img_dir = "data/processed_data/img"
    mask_dir = "data/processed_data/mask"
    
    # Load image and mask file paths
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

    # Ensure the number of images matches the number of masks
    assert len(img_files) == len(mask_files), "Mismatch between images and masks"
    
    
    # Define a dataset class for loading images and masks 
    class ImageMaskDataset(Dataset):
        def __init__(self, img_files, mask_files, transform=None):
            self.img_files = img_files
            self.mask_files = mask_files
            self.transform = transform
    
        def __len__(self):
            return len(self.img_files)
    
        def __getitem__(self, idx):
            try:
                img = cv2.imread(self.img_files[idx])[:, :, ::-1]  # Convert BGR to RGB
                mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            except Exception as e:
                raise RuntimeError(f"Error loading image or mask at index {idx}: {e}")
            
            img = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
              
            return img, mask
        
    
    # Create datasets
    dataset = ImageMaskDataset(img_files, mask_files, transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    
    # Initialize the model
    model = NeuralNetwork(e_channels, b_channels, d_channels, strides).to(device)
    
    # Print model size information
    total_params = sum(p.numel() for p in model.parameters()) # Total number of parameters in the model
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Loss function and optimizer
    # It is possible to use the pos_weight parameter to balance the loss function ######################################
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Example of inference
    model.eval()
    with torch.no_grad():
        # Use a sample from the validation dataset for inference
        example_input, actual_label = next(iter(val_loader))
        example_input = example_input[0].unsqueeze(0).to(device)  # Select the first sample and add batch dimension
        actual_label = actual_label[0].to(device)  # Ensure label is on the same device
        output = model(example_input)
        output = torch.sigmoid(output)  # Convert logits to probabilities
        predicted = (output > 0.5).float()  # Threshold to get binary mask

        # Plot the image and prediction
        try:
            plot_prediction(example_input[0], actual_label, predicted[0])
        except Exception as e:
            print(f"Could not plot image: {e}")


if __name__ == "__main__":
    main()
