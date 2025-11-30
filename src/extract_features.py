# src/extract_features.py
import os
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from skimage import filters, feature, io, color
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel, prewitt
import logging
from tqdm import tqdm

def extract_features_from_image(image_path):
    """Extract features from a single image using specified filters and GLCM properties."""
    try:
        # Load image as grayscale
        image = io.imread(image_path, as_gray=True)
        
        # Define angles for GLCM computation
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Initialize feature dictionary
        features = {'image_id': Path(image_path).stem, 'label': Path(image_path).parent.name}
        
        # Process each filter
        filters_to_apply = ['entropy', 'gaussian', 'sobel', 'gabor', 'hessian', 'prewitt']
        
        for filter_name in filters_to_apply:
            filtered_image = apply_filter(image, filter_name)
            
            # Compute GLCM for each angle and extract properties
            for i, angle in enumerate(angles):
                glcm = graycomatrix(filtered_image, [1], [angle], levels=256, symmetric=True, normed=True)
                
                # Extract GLCM properties
                properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
                for prop in properties:
                    feature_name = f"{filter_name}_{prop}_angle_{i}"
                    features[feature_name] = graycoprops(glcm, prop)[0, 0]
        
        return features
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def apply_filter(image, filter_name):
    """Apply specified filter to the image."""
    if filter_name == 'entropy':
        return filters.rank.entropy(image, footprint=np.ones((3, 3)))
    elif filter_name == 'gaussian':
        return filters.gaussian(image, sigma=1.0)
    elif filter_name == 'sobel':
        return sobel(image)
    elif filter_name == 'prewitt':
        return prewitt(image)
    elif filter_name == 'hessian':
        return filters.hessian(image, sigma=1.0)
    elif filter_name == 'gabor':
        # Use a single Gabor filter for simplicity
        gabor_filter = feature.gabor_filter(image, frequency=0.6, theta=0)[0]
        return np.abs(gabor_filter)
    else:
        return image  # Return original image if filter not recognized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="Input data path")
    parser.add_argument("--output_data", type=str, required=True, help="Output parquet file path")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Collect all image paths
    image_paths = list(Path(args.input_data).rglob("*.jpg"))
    image_paths.extend(Path(args.input_data).rglob("*.png"))
    
    logging.info(f"Found {len(image_paths)} images to process")
    
    # Extract features using multiprocessing
    with Pool() as pool:
        feature_list = list(tqdm(
            pool.imap(extract_features_from_image, image_paths),
            total=len(image_paths),
            desc="Extracting features"
        ))
    
    # Remove any failed extractions
    feature_list = [f for f in feature_list if f is not None]
    
    # Create DataFrame and save as Parquet
    df_features = pd.DataFrame(feature_list)
    df_features.to_parquet(args.output_data, index=False)
    
    end_time = time.time()
    
    # Log required metrics
    num_images = len(feature_list)
    num_features = len(df_features.columns) - 2  # Exclude image_id and label
    
    logging.info(f"Feature extraction completed successfully")
    logging.info(f"Number of images processed: {num_images}")
    logging.info(f"Number of features extracted per image: {num_features}")
    logging.info(f"Extraction time (seconds): {end_time - start_time:.2f}")
    logging.info(f"Compute SKU: {os.environ.get('AZUREML_COMPUTE', 'Unknown')}")
    
    print(f"Feature extraction completed. Processed {num_images} images with {num_features} features each.")
    print(f"Output saved to: {args.output_data}")

if __name__ == "__main__":
    main()   