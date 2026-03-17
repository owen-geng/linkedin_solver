import torch
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn,save,load
import numpy as np
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download= True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class digit_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit_net = self.cnn()

    def cnn(self):
        return nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*22*22,10)
        )
    
    def forward(self, x):
        x = self.digit_net(x)
        return x


if __name__ == "__main__":
    classifier = digit_classifier().to(device)
    optim = Adam(classifier.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    loss_list = []

    for epoch in range(10):
        for images,labels in train_loader:
            images,labels = images.to(device), labels.to(device)
            optim.zero_grad()
            outputs = classifier(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optim.step()
            loss_list.append(loss.item())

        print(f"Loss at Epoch {epoch} is {loss.item()}")

    torch.save(classifier.state_dict(), 'classifier.pt')
    
    plt.plot(loss_list)
    plt.show()