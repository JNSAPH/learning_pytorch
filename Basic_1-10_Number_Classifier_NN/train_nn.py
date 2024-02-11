# Pytorch
from torch import nn
from torch.optim import Adam # Optimizer
from torch.utils.data import DataLoader # Data Loader
import torch


# Torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Image Classifier
from ImageClassifierModel import ImageClassifier

# Get the data from the MNIST dataset and load it into a DataLoader to be used in the training loop
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)
    
# Instance of the model, loss function and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = ImageClassifier().to(device)
optimizer = Adam(model.parameters(), # Because we want to optimize the parameters of the model
                 lr=1e-3 # Learning rate
                 )
loss_fn = nn.CrossEntropyLoss()
epochs = 5

# Training Loop
if __name__ == "__main__":
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataset): 
            x, y = x.to(device), y.to(device) # Move the data to the device
            yhat = model(x) # Forward pass
            loss = loss_fn(yhat, y) # Calculate the loss

            # Apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} - Loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved")