import kagglehub
aryashah2k_indian_medicinal_leaves_dataset_path = kagglehub.dataset_download('aryashah2k/indian-medicinal-leaves-dataset')

print('Data source import complete.')

!pip install -U -q evaluate transformers datasets>=2.14.5 accelerate>=0.27 mlflow 2>/dev/null
!pip install transformers

!pip install --upgrade transformers

!pip install -U evaluate transformers datasets accelerate mlflow


import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,roc_auc_score,confusion_matrix,classification_report,f1_score)

import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (TrainingArguments,Trainer,ViTImageProcessor,ViTForImageClassification,DefaultDataCollator)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop,Compose,Normalize,RandomRotation,RandomResizedCrop,RandomHorizontalFlip,RandomAdjustSharpness,
Resize,ToTensor)

from PIL import ImageFile
# Loading images even if they are corrupted or incomplete.
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dict = {}
from pathlib import Path
from tqdm import tqdm
import os
MIN_SAMPLES = 100
file_names = []
labels = []
# Iterate through all image files in the specified directory
for file in sorted((Path('/kaggle/input/indian-medicinal-leaves-dataset/Indian Medicinal Leaves Image Datasets/').glob('*/*/*.jpg'))):
    # check number of such files in a directory
    sample_dir = '/'.join(str(file).split('/')[:-1])+'/'
    num_files_in_dir = [len(x) for _, _, x in os.walk(sample_dir)][0]
    if num_files_in_dir >= MIN_SAMPLES:
        file_names.append(str(file))
        label = str(file).split('/')[-2]
        labels.append(label)
print(len(file_names), len(labels), len(set(labels)))
dataset = Dataset.from_dict({"image": file_names, "label": labels}).cast_column("image", Image())
dataset[0]["image"]
labels_subset = labels[0]
print(labels_subset)

labels_list = ['Amla', 'Curry', 'Betel', 'Bamboo', 'Palak(Spinach)', 'Coriender', 'Ashoka', 'Seethapala', 'Lemon_grass', 'Pappaya', 'Curry_Leaf', 'Lemon', 'Nooni',
               'Henna', 'Mango', 'Doddpathre', 'Amruta_Balli', 'Betel_Nut', 'Tulsi', 'Pomegranate',
                'Castor', 'Jackfruit', 'Insulin', 'Pepper', 'Raktachandini', 'Aloevera', 'Jasmine', 'Doddapatre', 'Neem',
                'Geranium', 'Rose', 'Gauva', 'Hibiscus', 'Nithyapushpa', 'Wood_sorel', 'Tamarind', 'Guava', 'Bhrami', 'Sapota', 'Basale', 'Avacado',
               'Ashwagandha', 'Nagadali', 'Arali', 'Ekka', 'Ganike', 'Tulasi', 'Honge', 'Mint', 'Catharanthus', 'Papaya', 'Brahmi'] #list(set(labels))

label2id, id2label = dict(), dict()

for i, label in enumerate(labels_list):
    label2id[label] = i  # Map the label to its corresponding ID
    id2label[i] = label  # Map the ID to its corresponding label

print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)

ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

train_data = dataset['train']
test_data = dataset['test']

model_str = 'dima806/medicinal_plants_image_detection'
processor = ViTImageProcessor.from_pretrained(model_str)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
print("Size: ", size)
normalize = Normalize(mean=image_mean, std=image_std)

# set of transformations for training and validation data
_train_transforms = Compose([Resize((size, size)),RandomRotation(90),RandomAdjustSharpness(2),RandomHorizontalFlip(0.5),ToTensor(),normalize])
_val_transforms = Compose([Resize((size, size)),ToTensor(),normalize])

# function to apply training and validation transformations to a batch of examples
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples
def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Set the transforms for the training and test/validation data
train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)

# preparing batched data for model training.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

"""# Load, train, and evaluate model"""

model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))

model.config.id2label = id2label
model.config.label2id = label2id
print(model.num_parameters(only_trainable=True) / 1e6)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predicted_labels = predictions.argmax(axis=1)

    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {"accuracy": acc_score}

metric_name = "accuracy"
model_name = "medicinal_plants_image_detection"
num_train_epochs = 5

args = TrainingArguments(output_dir=model_name,logging_dir='./logs',evaluation_strategy="epoch",learning_rate=5e-7,per_device_train_batch_size=32,
                         per_device_eval_batch_size=8,num_train_epochs=num_train_epochs,weight_decay=0.02,warmup_steps=50,remove_unused_columns=False,
                         save_strategy='epoch',load_best_model_at_end=True,save_total_limit=1,report_to="none")

trainer = Trainer(model,args,train_dataset=train_data,eval_dataset=test_data,data_collator=collate_fn,compute_metrics=compute_metrics,
                  tokenizer=processor,)

trainer.evaluate()

trainer.train()

trainer.evaluate()

outputs = trainer.predict(test_data)
print(outputs.metrics)

y_true = outputs.label_ids
# Predict the labels by selecting the class with the highest probability
y_pred = outputs.predictions.argmax(1)
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
if len(labels_list) <= 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(22, 20))
print()
print("Classification report:")
print()
try:
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))
except:
    pass

trainer.save_model()

from transformers import pipeline
import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

pipe = pipeline('image-classification', model=model_name, device=0, use_fast=True)

image = test_data[6]["image"]
image

prediction = pipe(image)
print("Top 5 Predictions:")
for i, pred in enumerate(prediction):
    print(f"Rank {i+1}: Label = {pred['label']}, Confidence = {pred['score']:.4f}")

print("Final Model Prediction:", prediction[0]["label"])

true_label = id2label[test_data[6]["label"]]
print("True Label:", true_label)