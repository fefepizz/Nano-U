"""
Model Training Script for Nano-U with Cross-Validation

This script handles the training of neural network models for navigation tasks with k-fold cross-validation.
It supports both standard training and knowledge distillation, where a smaller
model learns from a larger pre-trained model.

Key features:
- K-fold cross-validation for robust model evaluation
- Standard training for BU_Net (teacher model)
- Knowledge distillation training for Nano_U (student model)
- Validation and metrics computation
- Model evaluation and visualization
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

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

def train_with_cross_validation(model_class, device, criterion, n_folds=5, epochs: int=1, learning_rate: float=1e-5, 
                       batch_size: int=1, teacher_model=None, distill_alpha=0.5, distill_temp=2.0, n_channels=3):
    """
    Train the model using k-fold cross-validation.
    
    Args:
        model_class: The class of the model to train
        device (str): Device to use ('cuda' or 'cpu')
        criterion: Loss function
        n_folds (int): Number of folds for cross-validation
        epochs (int): Number of training epochs per fold
        learning_rate (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        teacher_model (torch.nn.Module, optional): Teacher model for knowledge distillation
        distill_alpha (float): Weight for standard loss vs distillation loss
        distill_temp (float): Temperature parameter for knowledge distillation
        n_channels (int): Number of input channels for the model
        
    Returns:
        torch.nn.Module: The best trained model across all folds
    """
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Combine training and validation sets for cross-validation
    train_img_dir = "data/processed_data/train/img"
    train_mask_dir = "data/processed_data/train/mask"
    val_img_dir = "data/processed_data/val/img"
    val_mask_dir = "data/processed_data/val/mask"

    # Get all files from training and validation directories
    train_img_files = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if (f.endswith(".png"))])
    train_mask_files = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith(".png")])
    val_img_files = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if (f.endswith(".png"))])
    val_mask_files = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith(".png")])
    
    # Combine training and validation data for cross-validation
    all_img_files = train_img_files + val_img_files
    all_mask_files = train_mask_files + val_mask_files
    
    # Print information about the dataset
    print(f"Total dataset size: {len(all_img_files)} images")
    
    assert len(all_img_files) == len(all_mask_files), "Mismatch between images and masks"
    assert len(all_img_files) > 0, "No images found. Run data_preparation.py first."
    
    # Create the full dataset
    full_dataset = LoadDataset(all_img_files, all_mask_files, transform=transform)
    
    # Setup cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store best model and metrics
    best_model = None
    best_val_acc = 0.0
    all_fold_train_losses = []
    all_fold_train_accs = []
    all_fold_val_losses = []
    all_fold_val_accs = []
    
    # Train over all folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(full_dataset)))):
        print(f"\n{'='*80}\nFold {fold+1}/{n_folds}\n{'='*80}")
        
        # Create data loaders for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        
        print(f"Training set: {len(train_subset)} samples, Validation set: {len(val_subset)} samples")
        
        # Initialize a new model for each fold
        model = model_class(n_channels=n_channels)
        model = model.to(device, memory_format=torch.channels_last)
        
        if fold == 0:
            # Print model summary only for the first fold
            print("Model Summary:")
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {total_trainable_params:,}")
            print(f"Model size: {model_size_mb:.2f} MB")
        
        # Train the fold
        fold_model, fold_metrics = train_fold(model, device, criterion, train_loader, val_loader, 
                                              epochs, learning_rate, teacher_model, 
                                              distill_alpha, distill_temp, fold)
        
        # Unpack metrics
        train_losses, train_accs, val_losses, val_accs = fold_metrics
        
        # Store metrics for this fold
        all_fold_train_losses.append(train_losses)
        all_fold_train_accs.append(train_accs)
        all_fold_val_losses.append(val_losses)
        all_fold_val_accs.append(val_accs)
        
        # Check if this is the best model so far
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_model = fold_model
            # Save the best model
            torch.save(best_model.state_dict(), f'models/cv_best_model_fold_{fold+1}.pth')
            print(f"New best model saved (Fold {fold+1}, Accuracy: {best_val_acc:.4f})")
    
    # Compute average metrics across all folds
    avg_train_losses = np.mean([loss[-1] for loss in all_fold_train_losses])
    avg_train_accs = np.mean([acc[-1] for acc in all_fold_train_accs])
    avg_val_losses = np.mean([loss[-1] for loss in all_fold_val_losses])
    avg_val_accs = np.mean([acc[-1] for acc in all_fold_val_accs])
    
    print(f"\n{'='*80}")
    print(f"Cross-Validation Results ({n_folds} folds):")
    print(f"Average Train Loss: {avg_train_losses:.4f}")
    print(f"Average Train Accuracy: {avg_train_accs:.4f}")
    print(f"Average Validation Loss: {avg_val_losses:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accs:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*80}")
    
    # Plot average metrics
    avg_train_losses = np.mean(all_fold_train_losses, axis=0)
    avg_train_accs = np.mean(all_fold_train_accs, axis=0)
    avg_val_losses = np.mean(all_fold_val_losses, axis=0)
    avg_val_accs = np.mean(all_fold_val_accs, axis=0)
    
    plot_metrics(avg_train_losses, avg_val_losses, avg_train_accs, avg_val_accs, epochs, title="CV Average Metrics")
    plt.savefig("cv_metrics.png")
    plt.close()
    
    # Visualize a prediction using the best model
    best_model.eval()
    with torch.no_grad():
        # Create a validation loader with the full validation set for visualization
        val_dataset = LoadDataset(val_img_files, val_mask_files, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Get an example from the validation set for visualization
        try:
            example_input, example_label = next(iter(val_loader))
            if len(example_input) > 0:
                example_idx = min(2, len(example_input) - 1)  # Use index 2 if available, otherwise use the last one
                example_input = example_input[example_idx].unsqueeze(0).to(device)
                example_label = example_label[example_idx].to(device)
                
                output = best_model(example_input)
                output = torch.sigmoid(output)
                predicted = (output > 0.5).float()

                img_to_plot = example_input[0].clone().detach().cpu() * 0.5 + 0.5
                fig = plot_prediction(img_to_plot, example_label, predicted[0])
                plt.savefig("cv_prediction.png")
                plt.close(fig)
                print("Saved prediction visualization as cv_prediction.png")
        except Exception as e:
            print(f"Could not create prediction visualization: {e}")
    
    return best_model

def train_fold(model, device, criterion, train_loader, val_loader, epochs, learning_rate, 
               teacher_model=None, distill_alpha=0.5, distill_temp=2.0, fold_num=0):
    """
    Train a model for a single fold of cross-validation.
    
    Args:
        model (torch.nn.Module): The model to train
        device (str): Device to use ('cuda' or 'cpu')
        criterion: Loss function
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        teacher_model (torch.nn.Module, optional): Teacher model for knowledge distillation
        distill_alpha (float): Weight for standard loss vs distillation loss
        distill_temp (float): Temperature parameter for knowledge distillation
        fold_num (int): Current fold number for logging
        
    Returns:
        tuple: (trained_model, (train_losses, train_accs, val_losses, val_accs))
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm.tqdm(train_loader, desc=f"Fold {fold_num+1} - Epoch {epoch+1}/{epochs} (Train)", leave=True)

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
        
        train_loop.set_description(f"Fold {fold_num+1} - Epoch {epoch+1}/{epochs} (Train) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, epochs, fold_num)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Fold {fold_num+1} - Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save the model for this fold on the last epoch
        if epoch == epochs - 1:
            torch.save(model.state_dict(), f'models/cv_fold_{fold_num+1}_model.pth')
    
    # Return the trained model and the metrics
    return model, (train_losses, train_accs, val_losses, val_accs)

def validate(model, val_loader, criterion, device, epoch, epochs, fold_num=0):
    """
    Validate the model on the validation dataset.
    
    Args:
        model (torch.nn.Module): The model to validate
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function
        device (str): Device to use ('cuda' or 'cpu')
        epoch (int): Current epoch number
        epochs (int): Total number of epochs
        fold_num (int): Current fold number for logging
        
    Returns:
        tuple: (validation_loss, validation_accuracy)
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_loop = tqdm.tqdm(val_loader, desc=f"Fold {fold_num+1} - Epoch {epoch+1}/{epochs} (Validation)", leave=False)
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
        val_loop.set_description(f"Fold {fold_num+1} - Epoch {epoch+1}/{epochs} (Validation) Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    return val_loss, val_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set parameters for cross-validation
    n_folds = 5  # Number of folds for cross-validation
    epochs = 80  # Epochs per fold
    learning_rate = 1e-6
    batch_size = 8
    
    # Train BU_Net (teacher) with cross-validation
    print("Training BU_Net (teacher) model with cross-validation...")
    unet_criterion = nn.BCEWithLogitsLoss()
    trained_unet = train_with_cross_validation(
        model_class=BU_Net,
        device=device,
        criterion=unet_criterion,
        n_folds=n_folds,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_channels=3
    )
    
    # Save the final best model
    torch.save(trained_unet.state_dict(), 'models/BU_Net_cv.pth')
    print("BU_Net weights saved to models/BU_Net_cv.pth")

    """
    # Load Nano_U as student for distillation (after cross-validation)
    student_model = Nano_U(n_channels=3)
    student_model = student_model.to(device, memory_format=torch.channels_last)
    
    # Load BU_Net as teacher
    teacher_model = BU_Net(n_channels=3)
    teacher_weights_path = 'models/BU_Net_cv.pth'
    teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher_model = teacher_model.to(device, memory_format=torch.channels_last)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train Nano_U with distillation from BU_Net
    trained_model = train_with_cross_validation(
        model_class=Nano_U,
        device=device,
        criterion=criterion,
        n_folds=n_folds,
        epochs=30,
        learning_rate=1e-5,
        batch_size=8,
        teacher_model=teacher_model,
        distill_alpha=0.5,
        distill_temp=2.0,
        n_channels=3
    )
    
    torch.save(trained_model.state_dict(), 'models/Nano_U_distilled_cv.pth')
    print("Nano_U (student) weights saved to models/Nano_U_distilled_cv.pth")
    """
