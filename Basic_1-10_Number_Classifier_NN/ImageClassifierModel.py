# Pytorch
from torch import nn

# Image Classifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # Define the model
        self.model = nn.Sequential(
            nn.Conv2d(
                1, # input channels - because the images are in grayscale
                32, # output channels - number of filters
                kernel_size=3, # kernel size - 3x3 
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(), # Flatten the output of the previous layer because
            nn.Linear((64 * (28 - 6) * (28 - 6)), # Calculation is becuase image is 28x28 and we have 3 conv layers with kernel size 3
                      10 # Output layer
                      )

        )
    
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        return self.model(x)