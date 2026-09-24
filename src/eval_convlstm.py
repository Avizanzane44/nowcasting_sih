import os
import sys
import glob
import torch
import numpy as np
sys.path.append('models/convlstm')
from train import read_pgm_gz
from model import Seq2SeqConvLSTM

def generate_forecast():
    print("Generating ConvLSTM forecast for evaluation...")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "pysteps_data"))
    gz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*20160928*.pgm.gz"), recursive=True))
    
    if len(gz_files) < 15:
        print("Not enough frames to run evaluation.")
        return
        
    test_files = gz_files[-15:]
    input_paths = test_files[:3]
    
    # Load input frames
    input_frames = []
    for p in input_paths:
        img = read_pgm_gz(p)
        # We need to downsample for the model we trained
        img = img[::4, ::4]
        input_frames.append(img)
        
    input_seq = np.array(input_frames)
    input_seq = np.expand_dims(input_seq, axis=(0, 1)) # (batch=1, channels=1, seq=3, H, W)
    input_seq = np.swapaxes(input_seq, 1, 2) # (batch=1, seq=3, channels=1, H, W)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqConvLSTM(in_channels=1, hidden_channels=8, out_channels=1, kernel_size=(3, 3)).to(device)
    
    # Load weights if available
    weights_path = "convlstm_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained weights.")
    else:
        print("Weights not found, using untrained model.")
        
    model.eval()
    x = torch.tensor(input_seq, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(x, future_seq_len=12) # predict 12 frames
        
    preds = preds.squeeze().cpu().numpy() # (12, H, W)
    
    # We must upsample back to original size for fair comparison with ground truth
    import cv2
    original_shape = read_pgm_gz(input_paths[0]).shape # (H, W)
    upsampled_preds = []
    for i in range(preds.shape[0]):
        resized = cv2.resize(preds[i], (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized = resized * 50.0 
        upsampled_preds.append(resized)
        
    upsampled_preds = np.array(upsampled_preds)
    np.save("convlstm_forecast.npy", upsampled_preds)
    print("ConvLSTM predictions saved to convlstm_forecast.npy")

if __name__ == "__main__":
    generate_forecast()
