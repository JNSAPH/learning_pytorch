from ImageClassifierModel import ImageClassifier
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch import nn
from PIL import Image

model = ImageClassifier()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Test the model
if __name__ == "__main__":
    model.load_state_dict(torch.load("model.pth"))
    img = Image.open("test.png")
    img = img.resize((28, 28)) # Resize the image to 28x28
    img = img.convert("L") # Convert the image to grayscale
    img = ToTensor()(img).unsqueeze(0)
    img = img.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        print(model(img).argmax(1))