"""
Data Preparation Script for Nano-U

This script processes image and mask files from the TinyAgri dataset, resizes them, 
and organizes them into a structured format for model training.
The script:
1. Imports images and corresponding masks from different scenes
2. Resizes the images and masks to 64x48 pixels
3. Saves the processed data in an organized directory structure
"""

import os
import cv2
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
    
    # Get all mask files (for debugging)
    mask_files = sorted(glob(os.path.join(mask_folder, "*.png")))
    print(f"Found {len(mask_files)} masks in {mask_folder}")
    
    pairs = []
    
    # if the image has a corresponding mask, then process them
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
                    
                    # format the name as frame{i}_{j}.png
                    formatted_name = f"frame{folder_idx}_{frame_num}.png"
                    
                    # associate the image with the mask and formatted name
                    pairs.append((img_resized, mask_resized, formatted_name))
                    print(f"Processed image {frame_num}")
                    
                else:
                    print(f"something went wrong with {img_name}")
                
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


# put them in the same folder
for i, (img, mask, name) in enumerate(data):
    
    # name is formatted as frame{i}_{j}.png
    # remove the last 4 characters to get the base name (.png)
    base_name = os.path.join(output_dir, name[:-4])
    
    # Save the processed image
    img_path = os.path.join(output_dir, f"{base_name}.png")
    cv2.imwrite(img_path, img)
    
    # Save the processed mask
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask)


# Create img and mask subdirectories
img_dir = os.path.join(output_dir, "img")
mask_dir = os.path.join(output_dir, "mask")
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for i, (img, mask, name) in enumerate(data):
    # Save the processed image
    img_path = os.path.join(output_dir, "img", name)  # Use name directly
    cv2.imwrite(img_path, img)
    
    # Save the processed mask
    mask_name = name.replace(".png", "_mask.png")  # Append "_mask" to the name
    mask_path = os.path.join(output_dir, "mask", mask_name)
    cv2.imwrite(mask_path, mask)
    
    print(f"Saving image to: {img_path}")
    print(f"Saving mask to: {mask_path}")


print(f"Processed {len(data)} image-mask pairs and saved to {output_dir}")

# Add code to randomly divide the data into 70/20/10 splits
import random
import shutil

# Create split directories with img and mask subfolders
split_dirs = {
    "70": 0.7,  # 70% of the data
    "20": 0.2,  # 20% of the data
    "10": 0.1   # 10% of the data
}

# Create the split directories
for split_name in split_dirs.keys():
    split_dir = os.path.join(output_dir, split_name)
    
    # Create img and mask subdirectories
    split_img_dir = os.path.join(split_dir, "img")
    split_mask_dir = os.path.join(split_dir, "mask")
    if not os.path.exists(split_img_dir):
        os.makedirs(split_img_dir)
    if not os.path.exists(split_mask_dir):
        os.makedirs(split_mask_dir)

# Get all image file names (without paths)
all_img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
print(f"Total image files found: {len(all_img_files)}")

# Verify that each image has a corresponding mask
valid_pairs = []
for img_file in all_img_files:
    mask_file = img_file.replace(".png", "_mask.png")
    mask_path = os.path.join(mask_dir, mask_file)
    
    if os.path.exists(mask_path):
        valid_pairs.append((img_file, mask_file))
    else:
        print(f"Warning: Mask not found for image {img_file}, excluding from split")

print(f"Total valid image-mask pairs: {len(valid_pairs)}")

# Shuffle the pairs to randomize
random.shuffle(valid_pairs)

# Calculate the number of pairs for each split
total_pairs = len(valid_pairs)
split_counts = {
    split_name: int(percentage * total_pairs)
    for split_name, percentage in split_dirs.items()
}

# Adjust the counts if they don't sum up to total_pairs due to rounding
sum_counts = sum(split_counts.values())
if sum_counts < total_pairs:
    # Add the remaining pairs to the 70% split
    split_counts["70"] += (total_pairs - sum_counts)

# Keep track of current position in the shuffled list
current_idx = 0

# Copy files to their respective split directories
for split_name, count in split_counts.items():
    print(f"Copying {count} image-mask pairs to {split_name}% split")
    
    # Get the subset of pairs for this split
    split_pairs = valid_pairs[current_idx:current_idx + count]
    current_idx += count
    
    for img_file, mask_file in split_pairs:
        # Source paths
        src_img_path = os.path.join(img_dir, img_file)
        src_mask_path = os.path.join(mask_dir, mask_file)
        
        # Destination paths
        dst_img_path = os.path.join(output_dir, split_name, "img", img_file)
        dst_mask_path = os.path.join(output_dir, split_name, "mask", mask_file)
        
        # Copy the files
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)

print(f"Data split complete: 70% ({split_counts['70']} files), 20% ({split_counts['20']} files), 10% ({split_counts['10']} files)")