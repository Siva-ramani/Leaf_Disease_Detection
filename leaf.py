from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch
import json
import os
from Example import ResNet9  # Import the ResNet9 model

app = Flask(__name__)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(in_channels=3, num_diseases=38).to(device)
model.load_state_dict(torch.load("resnet9_model.pth", map_location=device))
model.eval()

# Load disease treatments from JSON file
with open("disease_treatments.json", "r") as f:
    disease_treatments = json.load(f)

# Define disease names globally
disease_names = list(disease_treatments["diseases"].keys())

# Define the function to preprocess the input image and make predictions
def predict_leaf_disease(image_path):
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    test_image = Image.open(image_path).convert('RGB')
    test_image = test_transform(test_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(test_image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Routes for the Flask app

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_path = "/Users/siva/Downloads/jancy/archive/Plant-Leaf-Disease-Prediction/static/images/uploaded_image.jpg"  # Set the correct file path
        file.save(file_path)
        predicted_class_index = predict_leaf_disease(file_path)
        predicted_class = disease_names[predicted_class_index]  # Convert index to class name
        treatment = disease_treatments["diseases"].get(predicted_class, "No treatment information available")
        return render_template('result.html', predicted_class=predicted_class, treatment=treatment)

@app.route("/result/<int:predicted_class_index>")
def result(predicted_class_index):
    # Render the result.html template with the predicted class index
    return render_template('result.html', predicted_class_index=predicted_class_index)

if __name__ == "__main__":
    app.run(threaded=False, port=8080, debug=True)
