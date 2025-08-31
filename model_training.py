"""
Model Training Script for Nano-U

This script handles the training of neural network models for navigation tasks.
It supports both standard training and knowledge distillation, where a smaller
model learns from a larger pre-trained model.

Key features:
- Standard training for BU_Net (teacher model)
- Knowledge distillation training for Nano_U (student model)
- Validation and metrics computation
- Model evaluation and visualization
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import tqdm
import matplotlib.pyplot as plt

from models.BU_Net import BU_Net
from models.Nano_U import Nano_U

from utils.LoadDataset import LoadDataset
from utils.metrics import plot_metrics, plot_prediction

def compute_loss(outputs, labels, criterion, distillation=False, teacher_outputs=None, alpha=0.5, temperature=2.0):
    """
    Compute the loss between outputs and labels using the provided criterion.
    If distillation is True, combine standard loss with distillation loss from teacher_outputs.
    alpha: weight for standard loss, (1-alpha) for distillation loss.
    temperature: softening parameter for distillation.
    """
    # Standard loss
    if not distillation or teacher_outputs is None:
        return criterion(outputs, labels)
    
    loss_stud = criterion(outputs, labels)
    # Distillation loss: MSE between student and teacher outputs
    mse_loss = torch.nn.MSELoss()
    loss_teacher = mse_loss(torch.sigmoid(outputs), torch.sigmoid(teacher_outputs))
    return alpha * loss_stud + (1 - alpha) * loss_teacher

def train(model, device, criterion, epochs: int=1, learning_rate: float=1e-5, batch_size: int=1, teacher_model=None, distill_alpha=0.5, distill_temp=2.0):
    """
    Train the model using the specified parameters.
    
    Args:
        model (torch.nn.Module): The model to train
        device (str): Device to use ('cuda' or 'cpu')
        criterion: Loss function
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        teacher_model (torch.nn.Module, optional): Teacher model for knowledge distillation
        distill_alpha (float): Weight for standard loss vs distillation loss
        distill_temp (float): Temperature parameter for knowledge distillation
        
    Returns:
        torch.nn.Module: The trained model
    """
    
    transform = transforms.Compose([
        # transforms.Resize((120, 120)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Use the 70/20 split directories for training and validation
    train_img_dir = "processed_data/70/img"
    train_mask_dir = "processed_data/70/mask"
    val_img_dir = "processed_data/20/img"
    val_mask_dir = "processed_data/20/mask"
    
    # Get all files from the training directory (70% split)
    train_img_files = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if (f.endswith(".png"))])
    train_mask_files = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith(".png")])
    
    # Get all files from the validation directory (20% split)
    val_img_files = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if (f.endswith(".png"))])
    val_mask_files = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith(".png")])

    # Print some information about the datasets
    print(f"Training set: {len(train_img_files)} images from {train_img_dir}")
    print(f"Validation set: {len(val_img_files)} images from {val_img_dir}")

    assert len(train_img_files) == len(train_mask_files), "Mismatch between training images and masks"
    assert len(val_img_files) == len(val_mask_files), "Mismatch between validation images and masks"
    assert len(train_img_files) > 0, f"No training images found in {train_img_dir}. Run data_preparation.py first."
    assert len(val_img_files) > 0, f"No validation images found in {val_img_dir}. Run data_preparation.py first."

    assert len(train_img_files) == len(train_mask_files), "Mismatch between training images and masks"
    assert len(val_img_files) == len(val_mask_files), "Mismatch between validation images and masks"

    train_dataset = LoadDataset(train_img_files, train_mask_files, transform=transform)
    val_dataset = LoadDataset(val_img_files, val_mask_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    
    model = model
    
    print("Model Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=True)

        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                outputs = model(inputs)
                loss = compute_loss(outputs, labels, criterion, distillation=True, teacher_outputs=teacher_outputs, alpha=distill_alpha, temperature=distill_temp)
            else:
                outputs = model(inputs)
                loss = compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5)
            labels_bin = (labels > 0.5)
            correct += (preds == labels_bin).sum().item()
            total += labels.numel()
       
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        train_loop.set_description(f"Epoch {epoch+1}/{epochs} (Train) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, epochs)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/{epochs}, 'f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc            
            torch.save(model.state_dict(), 'models/temp_model.pth')
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs, epochs)
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    model.eval()
    with torch.no_grad():
        # Get an example from the validation set for visualization
        try:
            example_input, example_label = next(iter(val_loader))
            if len(example_input) > 0:
                example_idx = min(2, len(example_input) - 1)  # Use index 2 if available, otherwise use the last one
                example_input = example_input[example_idx].unsqueeze(0).to(device)
                example_label = example_label[example_idx].to(device)
                
                output = model(example_input)
                output = torch.sigmoid(output)
                predicted = (output > 0.5).float()

                img_to_plot = example_input[0].clone().detach().cpu() * 0.5 + 0.5
                fig = plot_prediction(img_to_plot, example_label, predicted[0])
                plt.savefig("example_prediction.png")
                plt.close(fig)
                print("Saved prediction visualization as example_prediction.png")
        except Exception as e:
            print(f"Could not create prediction visualization: {e}")
    
    return model


def validate(model, val_loader, criterion, device, epoch, epochs):
    """
    Validate the model on the validation dataset.
    
    Args:
        model (torch.nn.Module): The model to validate
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function
        device (str): Device to use ('cuda' or 'cpu')
        epoch (int): Current epoch number
        epochs (int): Total number of epochs
        
    Returns:
        tuple: (validation_loss, validation_accuracy)
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)", leave=False)
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, labels, criterion)
            val_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5)
            labels_bin = (labels > 0.5)
            val_correct += (preds == labels_bin).sum().item()
            val_total += labels.numel()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_loop.set_description(f"Epoch {epoch+1}/{epochs} (Validation) Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    return val_loss, val_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    # Train BU_Net (teacher) from scratch and save weights
    unet_model = BU_Net(n_channels=3)
    unet_model = unet_model.to(device, memory_format=torch.channels_last)
    unet_criterion = nn.BCEWithLogitsLoss()
    print("Training BU_Net (teacher) model...")
    trained_unet = train(unet_model, device, unet_criterion, epochs=80, learning_rate=1e-6, batch_size=8)
    torch.save(trained_unet.state_dict(), 'models/BU_Net.pth')
    print("BU_Net weights saved to models/BU_Net.pth")

    """
    # Load Nano_U as student
    student_model = Nano_U(n_channels=3)
    student_model = student_model.to(device, memory_format=torch.channels_last)
    
    # Load BU_Net as teacher
    teacher_model = BU_Net(n_channels=3)
    teacher_weights_path = 'models/BU_Net.pth'
    teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher_model = teacher_model.to(device, memory_format=torch.channels_last)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train MU_Net with distillation from BU_Net
    trained_model = train(student_model, device, criterion, epochs=30, learning_rate=1e-5, batch_size=8, teacher_model=teacher_model, distill_alpha=0.5, distill_temp=2.0)
    torch.save(trained_model.state_dict(), 'models/MU_Net_distilled.pth')
    print("MU_Net (student) weights saved to models/MU_Net_distilled.pth")
    
