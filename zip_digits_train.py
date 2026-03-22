import torch
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from PIL import Image
import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt

IMG_SIZE = 28
EPOCHS = 10
BATCH_SIZE = 32

# --- Augmentation ---

SHIFTS = [(-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (-3, 3), (3, -3)]
KERNEL_SIZES = [(k, k) for k in range(1, 5)]  # (1,1) to (4,4)
PAD_AMOUNTS = [3, 6, 9]
ZOOM_PADS = [2, 6, 12, 20]  # padding around tight bbox before resize back — smaller = more zoomed in

def zoom_img(img, pad):
    """Crop tightly to the digit bounding box with `pad` pixels of margin, resize back."""
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (img.shape[1], img.shape[0]))

def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)

def dilate_img(img, ksize):
    return cv2.dilate(img, np.ones(ksize, np.uint8))

def erode_img(img, ksize):
    return cv2.erode(img, np.ones(ksize, np.uint8))

def pad_img(img, pad):
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (img.shape[1], img.shape[0]))

def expand(img):
    """Generate all augmented variants of a single numpy image."""
    variants = [img]

    # Single transforms
    for dx, dy in SHIFTS:
        variants.append(shift_img(img, dx, dy))
    for k in KERNEL_SIZES:
        variants.append(dilate_img(img, k))
        variants.append(erode_img(img, k))
    for p in PAD_AMOUNTS:
        variants.append(pad_img(img, p))
    for z in ZOOM_PADS:
        variants.append(zoom_img(img, z))

    # Shift + morph
    for dx, dy in SHIFTS:
        s = shift_img(img, dx, dy)
        for k in KERNEL_SIZES:
            variants.append(dilate_img(s, k))
            variants.append(erode_img(s, k))

    # Shift + pad
    for dx, dy in SHIFTS:
        s = shift_img(img, dx, dy)
        for p in PAD_AMOUNTS:
            variants.append(pad_img(s, p))

    # Morph + pad
    for k in KERNEL_SIZES:
        for p in PAD_AMOUNTS:
            variants.append(pad_img(dilate_img(img, k), p))
            variants.append(pad_img(erode_img(img, k), p))

    # Zoom + morph
    for z in ZOOM_PADS:
        zm = zoom_img(img, z)
        for k in KERNEL_SIZES:
            variants.append(dilate_img(zm, k))
            variants.append(erode_img(zm, k))

    # Shift + morph + pad
    for dx, dy in SHIFTS:
        s = shift_img(img, dx, dy)
        for k in KERNEL_SIZES:
            for p in PAD_AMOUNTS:
                variants.append(pad_img(dilate_img(s, k), p))
                variants.append(pad_img(erode_img(s, k), p))

    return variants

# --- Dataset ---

def get_label_from_filename(fname):
    match = re.match(r'^(\d+)', fname)
    return int(match.group(1)) if match else None

class ZipDigitDataset(Dataset):
    def __init__(self, folder, transform=None, split='train', seed=42):
        self.transform = transform
        self.samples = []

        fnames = sorted(f for f in os.listdir(folder) if f.endswith('.png'))
        labels_found = sorted(set(
            get_label_from_filename(f) for f in fnames
            if get_label_from_filename(f) is not None and get_label_from_filename(f) <= 9
        ))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels_found)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.num_classes = len(labels_found)

        all_samples = []
        for fname in fnames:
            label = get_label_from_filename(fname)
            if label is None or label > 9:
                continue
            img_np = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            idx = self.label_to_idx[label]
            for variant in expand(img_np):
                all_samples.append((variant, idx))

        rng = np.random.default_rng(seed)
        rng.shuffle(all_samples)
        mid = len(all_samples) // 2
        self.samples = all_samples[:mid] if split == 'train' else all_samples[mid:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, label = self.samples[idx]
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Model ---

class ZipDigitClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 7x7

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = ZipDigitDataset('digits_training', transform=transform, split='train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Classes: {dataset.idx_to_label}")
    print(f"Training samples: {len(dataset)}")

    model = ZipDigitClassifier(num_classes=dataset.num_classes).to(device)
    optim = Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    loss_list = []

    for epoch in range(EPOCHS):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optim.step()
            loss_list.append(loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}")

    torch.save({
        'model_state': model.state_dict(),
        'idx_to_label': dataset.idx_to_label,
        'num_classes': dataset.num_classes,
    }, 'zip.pt')
    print("Saved to zip.pt")

    plt.plot(loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
