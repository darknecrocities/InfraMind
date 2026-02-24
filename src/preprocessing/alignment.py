import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from src.utils.logging_utils import setup_logging

logger = setup_logging()

def histogram_matching(source, template):
    """
    Adjust the pixel values of a source image to match the histogram of a template image.
    """
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumulative sum of the counts
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate from t_values to s_values based on quantiles
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(old_shape).astype(np.uint8)

def align_images(image_t1, image_t2):
    """
    Align two images using ORB feature detection and matching.
    image_t1 and image_t2 can be PyTorch tensors or numpy arrays.
    Returns aligned_t1, aligned_t2 as numpy arrays.
    """
    # Convert tensors to numpy if necessary
    if isinstance(image_t1, torch.Tensor):
        image_t1 = (image_t1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if isinstance(image_t2, torch.Tensor):
        image_t2 = (image_t2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(image_t1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image_t2, cv2.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take top matches
    good_matches = matches[:int(len(matches) * 0.2)]
    
    if len(good_matches) < 4:
        logger.warning("Not enough matches to align images. Returning original images.")
        return image_t1, image_t2

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image_t1 to match image_t2
    height, width, channels = image_t2.shape
    aligned_t1 = cv2.warpPerspective(image_t1, h, (width, height))

    # Apply histogram matching to aligned_t1 to match image_t2
    matched_t1 = np.zeros_like(aligned_t1)
    for i in range(3): # For each channel
        matched_t1[:,:,i] = histogram_matching(aligned_t1[:,:,i], image_t2[:,:,i])

    logger.info("Successfully aligned images and matched histograms.")
    return matched_t1, image_t2

if __name__ == "__main__":
    # Test alignment
    import matplotlib.pyplot as plt
    from src.ingestion.loader import TimeSeriesDataset
    
    dataset = TimeSeriesDataset(data_root="data/raw")
    sample = dataset[0]
    if sample:
        img_t1 = sample['images'][0]
        img_t2 = sample['images'][1]
        
        aligned_t1, aligned_t2 = align_images(img_t1, img_t2)
        
        print(f"Aligned T1 shape: {aligned_t1.shape}")
        print(f"Aligned T2 shape: {aligned_t2.shape}")
