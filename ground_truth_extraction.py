import torch
import numpy as np
import cv2
import pandas as pd
import os
import re

from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print("CUDA is available:", torch.cuda.is_available())
        
def load_images(images_path, start_idx=0, end_idx=None):
    """
    Load and convert images from a directory to RGB format.
    
    Args:
        images_path (str): Path to the directory containing images
        start_idx (int, optional): Starting index for image selection. Defaults to 0.
        end_idx (int, optional): Ending index for image selection. Defaults to None (all images).
        
    Returns:
        list: List of loaded RGB images
    """
    def get_frame_number(filename):
        # Extract the frame number from filename like 'd5_s1_frame123.png'
        match = re.search(r'frame(\d+)', filename)
        return int(match.group(1)) if match else 0

    image_files = sorted(os.listdir(images_path), key=get_frame_number)
    if end_idx is None:
        end_idx = len(image_files)
    image_files = image_files[start_idx:end_idx]
    images = []
    for img_file in image_files:
        image_full_path = os.path.join(images_path, img_file)
        img = cv2.imread(image_full_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
    return images

def save_line_points_csv(line_points, csv_filepath):
    """
    Save extracted line points to a CSV file.
    
    Args:
        line_points (numpy.ndarray): Array containing vanishing point and sampled points
        csv_filepath (str): Path where to save the CSV file
    """
    columns = ['v_point'] + [f'point{i}' for i in range(1, 11)]
    df = pd.DataFrame(line_points, columns=columns)
    csv_dir = os.path.dirname(csv_filepath)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_filepath, index=False)

def process_mask(mask, mask_index, output_dir):
    """
    Process a single mask to find line points and save the mask.
    
    Args:
        mask (np.array): Binary mask array
        mask_index (int): Index of the current mask (1-based)
        output_dir (str): Directory where to save the processed mask
    
    Returns:
        tuple: (v_point, sampled_points) or (None, None) if no valid points found
    """
    height, width = mask.shape
    last_points = np.zeros(width, dtype=int)
    
    # Find the highest point (lowest y value) for each x coordinate
    for x in range(width):
        for y in range(height - 1):
            if mask[y][x] == 1:
                last_points[x] = y
                break
    
    min_val = height
    for val in last_points:
        if 0 < val < min_val:
            min_val = val
            
    # Save the mask regardless of validation for debugging
    os.makedirs(output_dir, exist_ok=True)
    mask_filepath = os.path.join(output_dir, f'mask{mask_index}.png')
    cv2.imwrite(mask_filepath, mask * 255)
    print(f"Mask {mask_index} saved to: {mask_filepath}")
    
    # Validate mask based on minimum value and mean
    # Stricter validation criteria
    mask_mean = np.mean(mask)
    if not (40 < min_val < height - 40 and mask_mean > 0.20):
        print(f"Mask {mask_index}: No valid points found (min_val={min_val}, mean={mask_mean:.3f})")
        return None, None
    
    # Find the median x coordinate for points at minimum y value
    indices = [i for i, val in enumerate(last_points) if val == min_val]
    if not indices:
        # If no indices match min_val, use the first non-zero value
        for x, val in enumerate(last_points):
            if val > 0:
                indices = [x]
                min_val = val
                break
    
    median_x = indices[len(indices) // 2] if indices else 0
    v_point = (int(median_x), int(min_val))
    print(f"Mask {mask_index}, v_point: {v_point}")
    
    # Sample points along the line
    SAMPLE_INTERVAL = 71  # Distance between sampled points
    sampled_points = [(x, int(last_points[x])) for x in range(0, width, SAMPLE_INTERVAL) if x < width]
    
    return v_point, sampled_points

def main():
    """
    Main function to process images and generate ground truth masks.
    
    This function:
    1. Loads images from the specified directory
    2. Initializes the SAM2 model
    3. Generates masks for each image
    4. Processes masks to extract line points
    5. Saves the results to a CSV file
    """
    # Load images
    ######################## CHANGE HERE BASED ON THE FOLDER ########################
    img_dir = os.path.join("data", "TinyAgri", "Tomatoes", "scene2")

    images = load_images(img_dir, start_idx=0)
    print(f"Loaded {len(images)} images from {img_dir}")

    # Initialize SAM2 model using Hugging Face
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the SAM2.1 large model
    sam = build_sam2_hf(model_id="facebook/sam2.1-hiera-large", device=device)
    
        # Configure mask generator with parameters suitable for ground segmentation
    # Using less strict parameters for better mask detection
    mask_gen = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=24,           # Keeping the same point density
        pred_iou_thresh=0.75,         # Lower threshold to allow more predictions
        stability_score_thresh=0.80,  # Lower threshold to include more masks
        stability_score_offset=0.85,  # Lower offset for better mask detection
        min_mask_region_area=4000     # Keeping the same minimum area
    )

    sam.to(device=device)
     
    # Generate masks for each image
    masks = []
    original_indices = []  # Track original image indices
    for img_idx, img in enumerate(images, start=1):  # Start indexing from 1
        anns = mask_gen.generate(img)
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        if len(sorted_anns) == 0:  # Simple check to avoid IndexError
            print(f"No masks found for image {img_idx}, skipping...")
            continue
        ann = sorted_anns[0]
        masks.append(ann['segmentation'].astype(np.uint8))
        original_indices.append(img_idx)  # Keep track of original image index
        print(f"Done with mask: {img_idx}")

    # Process masks and collect line points
    line_points = np.zeros((len(masks), 11), dtype=object)
    
    ############################## CHANGE ALSO HERE BASED ON THE FOLDER ###############################
    output_dir = os.path.join('data', 'masks', 'Tomatoes', 'scene2')

    for mask_idx, (mask, orig_idx) in enumerate(zip(masks, original_indices)):
        v_point, sampled_points = process_mask(mask, orig_idx, output_dir)  # Use original index for naming
        if v_point and sampled_points:
            line_points[mask_idx, 0] = v_point  # Use mask_idx for array indexing
            for point_idx, point in enumerate(sampled_points):
                line_points[mask_idx, point_idx + 1] = point

    # Save results
    ############################## CHANGE ALSO HERE BASED ON THE FOLDER ###############################
    csv_filepath = os.path.join("data", "csv", "line_points_ts2.csv")
    save_line_points_csv(line_points, csv_filepath)

if __name__ == "__main__":
    main()