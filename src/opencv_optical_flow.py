import cv2
import numpy as np

def extrapolate_opencv(frames, num_lead_times):
    """
    Given a list of recent radar frames (shape H, W), compute optical flow
    between the last two frames, and extrapolate `num_lead_times` steps into the future.
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for optical flow")
    
    # Scale for OpenCV (0-255 uint8 works best for Farneback)
    img1 = np.clip(frames[-2] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    img2 = np.clip(frames[-1] * (255.0 / 50.0), 0, 255).astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    h, w = frames[-1].shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    forecasts = []
    current_frame = frames[-1]
    
    for step in range(1, num_lead_times + 1):
        # We want to warp current_frame using the flow
        # In backwards warping, we find where each pixel in the NEW frame comes from in the OLD frame.
        # If flow vector is (dx, dy), then new_pos = old_pos + dx -> old_pos = new_pos - dx
        # Since we are stepping further into the future, we multiply the flow by the step
        
        map_x = np.float32(grid_x - flow[:, :, 0] * step)
        map_y = np.float32(grid_y - flow[:, :, 1] * step)
        
        # Remap
        forecast = cv2.remap(frames[-1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        forecasts.append(forecast)
        
    return np.array(forecasts)
