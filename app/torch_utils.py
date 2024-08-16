import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io

#load model
#image -> tensor
#predict



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
#hyperparameter
input_size = 784
hidden_size = 100
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

PATH = "app/mnist_ffn.pth"

model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor()])
                                    #transforms.Normalize((0.1307,), (0.3081, ))
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_predictions(image_tensor):
    image_tensor = image_tensor.reshape(-1, 28*28)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data,1)
    return predicted