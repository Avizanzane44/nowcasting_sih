import os
import glob
import gzip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

from model import Seq2SeqConvLSTM

def read_pgm_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f"Not a P5 PGM file: {file_path}")
        while True:
            line = f.readline()
            if not line.startswith(b'#'): break
        dims = line.split()
        width, height = int(dims[0]), int(dims[1])
        maxval = int(f.readline().strip())
        img_data = f.read()
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
        img = img.astype(np.float32) / 255.0
        return img

class RadarSequenceDataset(Dataset):
    def __init__(self, file_paths, input_frames=3, target_frames=12, downsample=4, augment=False):
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.total_seq_len = input_frames + target_frames
        self.augment = augment
        
        self.frames = []
        for p in file_paths:
            img = read_pgm_gz(p)
            img = img[::downsample, ::downsample]
            self.frames.append(img)
            
        self.frames = np.array(self.frames)
        self.num_sequences = len(self.frames) - self.total_seq_len + 1
        
    def __len__(self):
        return max(0, self.num_sequences)

    def __getitem__(self, idx):
        seq = self.frames[idx : idx + self.total_seq_len]
        seq = np.expand_dims(seq, axis=1) # (seq, 1, H, W)
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        
        # Data Augmentation (spatial)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                seq_tensor = TF.hflip(seq_tensor)
            # Random vertical flip
            if random.random() > 0.5:
                seq_tensor = TF.vflip(seq_tensor)
            # Random rotation (0, 90, 180, 270)
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                seq_tensor = TF.rotate(seq_tensor, angle)
            # Note: cropping could be added, but keeping it simple for now as shapes must match model eval.
            
        x = seq_tensor[:self.input_frames]
        y = seq_tensor[self.input_frames:]
        return x, y

def train():
    print("Starting ConvLSTM Training Pipeline...")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pysteps_data"))
    gz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*20160928*.pgm.gz"), recursive=True))
    
    if len(gz_files) < 15:
        print(f"ERROR: Dataset is too small ({len(gz_files)} frames).")
        return
        
    # Strictly hold out the last 15 frames as a completely unseen TEST set
    test_size = 15
    train_val_files = gz_files[:-test_size]
    test_files = gz_files[-test_size:]
    
    print(f"Total frames: {len(gz_files)} | Train/Val: {len(train_val_files)} | Held-out Test: {len(test_files)}")
    
    # Synthetically expand training set via augmentation
    dataset = RadarSequenceDataset(train_val_files, input_frames=3, target_frames=12, downsample=4, augment=True)
    
    if len(dataset) == 0:
        print("Not enough sequences to train.")
        return
        
    val_split = int(len(dataset) * 0.2)
    if val_split == 0: val_split = 1
    train_size = len(dataset) - val_split
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_split])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqConvLSTM(in_channels=1, hidden_channels=8, out_channels=1, kernel_size=(3, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 4
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x, future_seq_len=y.shape[1])
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x, future_seq_len=y.shape[1])
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}")
        
    torch.save(model.state_dict(), "convlstm_weights.pth")
    print("Training complete. Weights saved to convlstm_weights.pth")

if __name__ == "__main__":
    train()
