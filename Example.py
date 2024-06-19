import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# from Training import ResNet9
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the convolutional block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Define the ResNet9 architecture
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Use AdaptiveAvgPool2d instead of MaxPool2d
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# Define the SimpleResidualBlock, accuracy function, ImageClassificationBase, ConvBlock, and ResNet9 classes here...

# Load dataset
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Assuming you have a single image for testing named "test_image.jpg"
test_image_path = "/Users/siva/Downloads/jancy/archive/Plant-Leaf-Disease-Prediction/Dataset/test/PotatoHealthy1.JPG"
test_image = Image.open(test_image_path).convert('RGB')
test_image = train_transform(test_image).unsqueeze(0)

# Define the disease names
disease_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
     'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
       'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
          'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
             'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(in_channels=3, num_diseases=38).to(device)

# Load trained model state
model.load_state_dict(torch.load("resnet9_model.pth", map_location=device))

# Evaluate the model
def test_model(model, test_image):
    model.eval()
    with torch.no_grad():
        test_image = test_image.to(device)
        output = model(test_image)
        _, predicted = torch.max(output, 1)
        predicted_class = disease_names[predicted.item()]
        print("Predicted class:", predicted_class)

# Test the model with one input image
test_model(model, test_image)

# # Calculate accuracy
# def calculate_accuracy(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / total
#     return accuracy

# Load test dataset
test_data = datasets.ImageFolder(root='/Users/siva/Downloads/jancy/archive/Plant-Leaf-Disease-Prediction/Dataset/val', transform=train_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# # Print accuracy
# test_accuracy = calculate_accuracy(model, test_loader)
# print("Test Accuracy:", test_accuracy)
