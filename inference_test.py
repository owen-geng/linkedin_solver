import torch
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from digit_classifier_train import digit_classifier
import cv2 as cv
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = digit_classifier().to(device)

with open('classifier.pt','rb') as f:
    classifier.load_state_dict(load(f))

img = cv.imread("digit1.png", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (28,28))
img_tensor = torch.tensor(img, dtype=torch.float32)/255.0
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
print(img_tensor.size())
cv.imshow("",img)
cv.waitKey(0)

start = time.time()
output = classifier(img_tensor)
end = time.time()
print(end-start)
print(output)
print(torch.argmax(output,dim=1))
