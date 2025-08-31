"""
Data Preparation Script for Nano-U

This script processes image and mask files from the TinyAgri dataset, resizes them, 
and organizes them into a structured format for model training.
The script:
1. Imports images and corresponding masks from different scenes
2. Resizes the images and masks to 64x48 pixels
3. Maintains strict image-mask pair associations throughout processing
4. Randomly divides the data into 70/20/10 splits for training/validation/testing
5. Ensures consistent naming and pairing in the output directory structure
"""

import os
import cv2
import random
from glob import glob


# import all the images from the folders
f1 = os.path.join("data/TinyAgri/Tomatoes/", "scene1")
f2 = os.path.join("data/TinyAgri/Tomatoes/", "scene2")
f3 = os.path.join("data/TinyAgri/Crops/", "scene1")
f4 = os.path.join("data/TinyAgri/Crops/", "scene2")

# import all the masks from the folders
m1 = os.path.join("data/masks/", "Tomatoes/scene1")
m2 = os.path.join("data/masks/", "Tomatoes/scene2")
m3 = os.path.join("data/masks/", "Crops/scene1")
m4 = os.path.join("data/masks/", "Crops/scene2")

# Create output directory
output_dir = "data/processed_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process images and masks
def process_data(img_folder, mask_folder, folder_idx):
    """
    Process images and masks from the specified folders.
    
    Args:
        img_folder (str): Path to the folder containing images
        mask_folder (str): Path to the folder containing masks
        folder_idx (int): Index identifier for the folder/scene
        
    Returns:
        list: List of tuples containing (resized_image, resized_mask, formatted_name)
    """
    
    # Get all image files
    img_files = sorted(glob(os.path.join(img_folder, "*.png")))
    print(f"Found {len(img_files)} images in {img_folder}")
    
    pairs = []
    processed_count = 0
    
    # Process each image if it has a corresponding mask
    for img_path in img_files:
        # Get the image name
        img_name = os.path.basename(img_path) 
        
        try:
            # This line extracts the frame number from the image filename
            # Split at 'frame' and take the second part 
            # then split at '.' to get the number (first part)
            frame_num = int(img_name.split('frame')[1].split('.')[0])
            
            # Construct the corresponding mask name
            mask_name = f"mask{frame_num}.png"
            mask_path = os.path.join(mask_folder, mask_name)
            
            # Check if the mask exists
            if os.path.exists(mask_path):
                
                # Read image and mask
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    
                    # reshape the image from 640x480
                    img_resized = cv2.resize(img, (64, 48))
                    
                    # reshape the mask from 640x480
                    mask_resized = cv2.resize(mask, (64, 48))
                    
                    # format the name with a consistent pattern that includes folder index and sequence number
                    # to ensure unique and consistent pairing
                    formatted_name = f"frame{folder_idx}_{processed_count:04d}.png"
                    
                    # associate the image with the mask and formatted name
                    pairs.append((img_resized, mask_resized, formatted_name))
                    processed_count += 1
                    
                    # Print progress for every 10th image or the last one
                    if processed_count % 10 == 0 or processed_count == len(img_files):
                        print(f"Processed {processed_count}/{len(img_files)} images from {img_folder}")
                    
                else:
                    print(f"Failed to read image or mask for {img_name}")
                
            else:
                print(f"Mask not found for image {frame_num}")
                
        except Exception as e:
            print(f"Could not process {img_name}: {e}")
    
    return pairs

# Process all folders
data = []
data.extend(process_data(f1, m1, 1))
data.extend(process_data(f2, m2, 2))
data.extend(process_data(f3, m3, 3))
data.extend(process_data(f4, m4, 4))

print(f"Total processed image-mask pairs: {len(data)}")

# Shuffle the data to randomize before splitting
random.shuffle(data)

# Define splits
split_dirs = {
    "train": 0.7,  # 70% of the data for training
    "val": 0.2,    # 20% of the data for validation
    "test": 0.1    # 10% of the data for testing
}

# Calculate number of samples for each split
total_samples = len(data)
split_counts = {
    split_name: int(percentage * total_samples)
    for split_name, percentage in split_dirs.items()
}

# Adjust counts if they don't sum up to total_samples due to rounding
sum_counts = sum(split_counts.values())
if sum_counts < total_samples:
    # Add the remaining pairs to the training split
    split_counts["train"] += (total_samples - sum_counts)

    # Create directories for each split
for split_name in split_dirs.keys():
    # Create img and mask subdirectories in one step
    os.makedirs(os.path.join(output_dir, split_name, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split_name, "mask"), exist_ok=True)# Partition the data into splits
current_idx = 0
processed_count = 0
total_count = sum(split_counts.values())

for split_name, count in split_counts.items():
    print(f"Allocating {count} image-mask pairs to {split_name} split")
    
    # Get the subset for this split
    split_data = data[current_idx:current_idx + count]
    current_idx += count
    
    # Save the image-mask pairs for this split
    for img, mask, name in split_data:
        # Save image and mask with consistent naming
        cv2.imwrite(os.path.join(output_dir, split_name, "img", name), img)
        cv2.imwrite(os.path.join(output_dir, split_name, "mask", name.replace(".png", "_mask.png")), mask)
        
        processed_count += 1
        # Print progress periodically
        if processed_count % 20 == 0 or processed_count == total_count:
            print(f"Progress: {processed_count}/{total_count} pairs saved ({(processed_count/total_count)*100:.1f}%)")

# Create a summary of the splits
print("\nData split summary:")
for split_name, count in split_counts.items():
    img_count = len(os.listdir(os.path.join(output_dir, split_name, "img")))
    mask_count = len(os.listdir(os.path.join(output_dir, split_name, "mask")))
    print(f"{split_name}: {count} pairs ({img_count} images, {mask_count} masks)")

print("\nData preparation complete!")