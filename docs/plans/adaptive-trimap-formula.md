import cv2
import numpy as np

def generate_adaptive_trimap(sam_mask, depth_map):
    # 1. Calculate the "Messiness" (Gradient Magnitude) of the Depth Map
    # This acts as our "dynamicism" factor.
    # Hair will have high gradient values; Shoulders/Walls will have low values.
    gX = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    gY = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gX, gY)
    
    # Normalize magnitude to 0 - 1 range for easier math
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Blur the magnitude slightly to create a "field of influence" 
    # (so a hair strand expands the area around it)
    complexity_map = cv2.GaussianBlur(magnitude, (21, 21), 0)

    # 2. Calculate Distance from the Core Body
    # Invert mask: 0 is body, 255 is background
    inv_mask = cv2.bitwise_not(sam_mask)
    # dist_transform: Each pixel's value is its distance (in pixels) from the body
    dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

    # 3. The Dynamic Formula
    # "Max Reach" is the absolute limit (e.g., 60px) purely for performance.
    # We multiply the Complexity Map by the Max Reach.
    # If complexity is 0 (smooth shoulder), reach is 0px.
    # If complexity is 1 (crazy hair), reach is 60px.
    dynamic_threshold = 2.0 + (60.0 * complexity_map) # Base 2px + up to 60px dynamic
    
    # 4. Create the Trimap
    # Logic: If pixel distance is LESS than its allowed dynamic threshold, it's Unknown.
    unknown_zone = (dist_map < dynamic_threshold) & (dist_map > 0)
    
    # Construct Final
    trimap = np.zeros_like(sam_mask)
    trimap[sam_mask == 255] = 255       # Core (White)
    trimap[unknown_zone] = 128          # Adaptive Unknown (Gray)
    
    return trimap