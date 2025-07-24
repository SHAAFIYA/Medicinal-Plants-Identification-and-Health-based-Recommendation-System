import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # ✅ Enable CORS properly

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
MODEL_DIR = "Medicinal_Plant_Model"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.safetensors")

# Load model configuration
config = AutoConfig.from_pretrained(MODEL_DIR)

# Load model architecture
model = AutoModelForImageClassification.from_config(config)

# Load weights
model.load_state_dict(load_file(MODEL_WEIGHTS))
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Secure filename & save
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Predict plant name
    predicted_class, confidence = predict_plant(file_path)

    return jsonify({
        "filename": filename,
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    })

def predict_plant(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)  
        confidence, predicted_class = torch.max(probabilities, 1)  

    class_labels = [
        'Amla', 'Curry', 'Betel', 'Bamboo', 'Palak(Spinach)', 'Coriender', 'Ashoka',
        'Seethapala', 'Lemon_grass', 'Pappaya', 'Curry_Leaf', 'Lemon', 'Nooni',
        'Henna', 'Mango', 'Doddpathre', 'Amruta_Balli', 'Betel_Nut', 'Tulsi', 'Pomegranate',
        'Castor', 'Jackfruit', 'Insulin', 'Pepper', 'Raktachandini', 'Aloevera', 'Jasmine',
        'Doddapatre', 'Neem', 'Geranium', 'Rose', 'Gauva', 'Hibiscus', 'Nithyapushpa',
        'Wood_sorel', 'Tamarind', 'Guava', 'Bhrami', 'Sapota', 'Basale', 'Avacado',
        'Ashwagandha', 'Nagadali', 'Arali', 'Ekka', 'Ganike', 'Tulasi', 'Honge', 'Mint',
        'Catharanthus', 'Papaya', 'Brahmi'
    ]

    return class_labels[predicted_class.item()], confidence.item() * 100

if __name__ == "__main__":
    app.run(debug=True)
