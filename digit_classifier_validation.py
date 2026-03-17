import torch
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn,save,load
import numpy as np
from matplotlib import pyplot as plt
from digit_classifier_train import digit_classifier


transform = transforms.Compose([transforms.ToTensor()])
val_dataset = datasets.MNIST(root="data", download= True, train=False, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = digit_classifier().to(device)
with open('classifier.pt','rb') as f:
    classifier.load_state_dict(load(f))

correct=0
total=0

with torch.no_grad():
    for images, labels in val_loader:
        images,labels = images.to(device), labels.to(device)

        out = classifier(images)
        pred= torch.argmax(out, dim=1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

accuracy = correct/total
print(accuracy)


