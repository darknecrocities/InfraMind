import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_synthetic_data(data_root, num_locations=5, img_size=(512, 512)):
    """
    Generates synthetic multi-temporal images with simulated crack growth.
    """
    os.makedirs(data_root, exist_ok=True)
    
    for i in range(num_locations):
        loc_id = f"LOC_{i:03d}"
        loc_path = os.path.join(data_root, loc_id)
        os.makedirs(loc_path, exist_ok=True)
        
        # Base image (concrete texture simulation)
        base_img = np.random.randint(180, 220, (img_size[1], img_size[0], 3), dtype=np.uint8)
        
        # Generate 3 timesteps
        for t in range(3):
            img = base_img.copy()
            # Simulate a crack growing
            pts = []
            start_x, start_y = img_size[0]//2, img_size[1]//2
            curr_x, curr_y = start_x, start_y
            
            # Growth parameters
            intensity = 20 * (t + 1)
            length = 50 * (t + 1)
            
            for _ in range(length):
                pts.append((curr_x, curr_y))
                curr_x += random.randint(-2, 2)
                curr_y += random.randint(1, 3)
            
            # Draw crack
            for p_idx in range(len(pts)-1):
                cv2.line(img, pts[p_idx], pts[p_idx+1], (50, 50, 50), thickness=2)
            
            # Save image
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(loc_path, f"2023-0{t+1}-01.jpg"))
            
    print(f"Generated synthetic data for {num_locations} locations in {data_root}")

if __name__ == "__main__":
    generate_synthetic_data("data/raw")
