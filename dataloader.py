import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class SurfaceDefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(f"{self.img_dir}/{self.imgs[idx]}", 0)
        mask = cv2.imread(f"{self.mask_dir}/{self.imgs[idx]}", 0)

        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)

        return img, mask

def create_data_loader():
    dataset = SurfaceDefectDataset("data/images", "data/masks")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataset, loader
