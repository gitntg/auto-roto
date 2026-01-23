import cv2
import numpy as np

def get_motion_factor(prev_frame_gray, curr_frame_gray):
    # 1. Calculate Dense Optical Flow (Farneback Algorithm)
    # This gives us a Vector (dx, dy) for every pixel
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray, curr_frame_gray, None, 
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # 2. Convert Vectors to Magnitude (Speed)
    # Mag = sqrt(dx^2 + dy^2)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 3. Normalize for our formula
    # Motion blur usually caps out visually around 20-30 pixels of speed.
    # We clip it so crazy camera pans don't explode the matte.
    mag = np.clip(mag, 0, 30)
    
    # Normalize to 0.0 - 1.0 range
    motion_factor = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    
    # 4. Expand the influence
    # Motion blur extends BEHIND the movement. We blur the map slightly
    # to ensure the trimap covers the trail, not just the current pixel.
    motion_factor = cv2.GaussianBlur(motion_factor, (15, 15), 0)
    
    return motion_factor