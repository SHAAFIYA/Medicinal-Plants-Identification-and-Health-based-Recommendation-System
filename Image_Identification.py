!pip install transformers torch torchvision

!nvidia-smi

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("/content/drive/MyDrive/Medicinal_Plant_Model/",
    use_fast=True)

model = AutoModelForImageClassification.from_pretrained(
    "/content/drive/MyDrive/Medicinal_Plant_Model",
    use_safetensors=True
)

from PIL import Image

image_path = "/content/drive/MyDrive/Test_Images/Bamboo.jpg"
image = Image.open(image_path)
display(image)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class = logits.argmax(-1).item()
predicted_plant = model.config.id2label[predicted_class]

print(f"\n Predicted Plant: {predicted_plant}")

import json

benefits_file_path = "/content/drive/MyDrive/benefits.json"

with open(benefits_file_path, "r", encoding="utf-8") as file:
    benefits_data = json.load(file)

matching_key = None

for key in benefits_data.keys():
    if predicted_plant in key:
        matching_key = key
        break
if matching_key:
    name = matching_key
    benefits = benefits_data[matching_key]["benefits"]

    print(f"Name: {name}")
    print("Medicinal Benefits:")
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
else:
    print("No medicinal benefits found for this plant.")