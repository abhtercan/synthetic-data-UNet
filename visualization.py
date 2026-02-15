import matplotlib.pyplot as plt
import torch
import cv2


def train_sample_visualize():
    sample_img = cv2.imread("data/images/0.png", 0)
    sample_mask = cv2.imread("data/masks/0.png", 0)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Synthetic Surface")
    plt.imshow(sample_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Ground Truth Mask")
    plt.imshow(sample_mask, cmap="gray")
    plt.axis("off")

    plt.show()


def model_output_visualize(model, dataset, device):
    model.eval()
    with torch.no_grad():
        img, mask = dataset[3]
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred > 0.5, cmap="gray")
    plt.axis("off")

    plt.show()