import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pretrained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Put the model in evaluation mode
model.eval()

# Image preprocessing required by ResNet
transform = models.ResNet18_Weights.DEFAULT.transforms()

# Sample image files
image_files = [
    "images/cat.jpg",
    "images/dog.jpg",
    "images/car.jpg"
]

# ImageNet class labels
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

for image_file in image_files:
    # Open image
    image = Image.open(image_file).convert("RGB")

    # Prepare image for the model
    input_image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_image)

    # Find the class with the highest score
    predicted_class = output.argmax(1).item()

    print(image_file, "->", labels[predicted_class])

    # Display the image
    plt.imshow(image)
    plt.title(labels[predicted_class])
    plt.axis("off")
    plt.show()