import cv2
import os
import time
import json
import sys
from datetime import datetime
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

output_folder = "captured_images"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

latest_image_path = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Press SPACE to capture | Press 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_image_path = os.path.join(output_folder, f"captured_{timestamp}.jpg")
        cv2.imwrite(latest_image_path, frame)
        print(f"Image saved: {latest_image_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if latest_image_path is None:
    print("No image captured. Exiting.")
    exit()

model_path = "Medicinal_Plant_Model"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(latest_image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

labels = model.config.id2label
predicted_plant = labels[predicted_class]

print(f"\nPredicted Plant: {predicted_plant}")


benefits_json_path = "benefits.json"

if os.path.exists(benefits_json_path):
    with open(benefits_json_path, "r", encoding="utf-8") as f:
        benefits_data = json.load(f)
else:
    print("benefits.json not found! Ensure the file is in the correct location.")
    exit()

matched_plant = None
for plant_name in benefits_data.keys():
    if predicted_plant in plant_name:
        matched_plant = plant_name
        break

if matched_plant:
    benefits = benefits_data[matched_plant]["benefits"]

    print("Medicinal Benefits:")
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
else:
    print("No medicinal benefits found for this plant.")