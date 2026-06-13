
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# ===== DATASET =====
class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.filenames = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        img = img.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# ===== MODEL BLOKLARI =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1))
        self.final = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        size = x.shape[2:]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x5 = F.relu(F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=True))
        return F.relu(self.bn(self.final(torch.cat([x1, x2, x3, x4, x5], dim=1))))

# ===== ANA MODEL =====
class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, features[0])
        self.enc2 = ResidualBlock(features[0], features[1])
        self.enc3 = ResidualBlock(features[1], features[2])
        self.enc4 = ResidualBlock(features[2], features[3])
        self.se1 = SEBlock(features[0])
        self.se2 = SEBlock(features[1])
        self.se3 = SEBlock(features[2])
        self.se4 = SEBlock(features[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ASPP(features[3], features[3] * 2)
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.dec4 = ResidualBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = ResidualBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = ResidualBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = ResidualBlock(features[0] * 2, features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        e1 = self.se1(self.enc1(x))
        e2 = self.se2(self.enc2(self.pool(e1)))
        e3 = self.se3(self.enc3(self.pool(e2)))
        e4 = self.se4(self.enc4(self.pool(e3)))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final_conv(d1))

# ===== LOSS =====
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

# ===== TRAINING =====
def train(image_dir, mask_dir, epochs=50, lr=0.0001, save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    dataset = PolypDataset(image_dir, mask_dir, augment=False)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_ds.dataset.augment = True
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    
    model = ResUNetPlusPlus().to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_dice = 0.0
    patience, counter = 10, 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                pred_bin = (preds > 0.5).float()
                intersection = (pred_bin * masks).sum()
                val_dice += (2 * intersection / (pred_bin.sum() + masks.sum() + 1e-6)).item()
        
        tl = train_loss / len(train_loader)
        vl = val_loss / len(val_loader)
        vd = val_dice / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val Dice: {vd:.4f}")
        
        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved! (Dice: {best_dice:.4f})")
        
        if vl < patience:
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
    
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    return model

if __name__ == "__main__":
    train(
        image_dir="data/PNG/Original",
        mask_dir="data/PNG/Ground Truth"
    )
