import cv2
from data_create import generate_surface
from dataloader import create_data_loader
from visualization import train_sample_visualize, model_output_visualize
from model import UNet
import torch.nn as nn
import torch
from train import train_model


NUM_SAMPLES = 800

for i in range(NUM_SAMPLES):
    img, msk = generate_surface()
    cv2.imwrite(f"data/images/{i}.png", img)
    cv2.imwrite(f"data/masks/{i}.png", msk * 255)


train_sample_visualize()


dataset, loader = create_data_loader()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
lossFunction = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 10

train_model(model, loader, device, lossFunction, optimizer, EPOCHS)

model_output_visualize(model, dataset, device)