#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().system('pip install mplcyberpunk')


# In[31]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcyberpunk
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from warnings import filterwarnings


# In[4]:


DATASET_PATH = '/kaggle/input' 

# Function to load files from the dataset directory
def load_dataset(path):
    file_list = []  # List to store file paths
    try:
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                # Append full path to the file list
                file_list.append(os.path.join(dirname, filename))
    except Exception as e:
        print(f"Error accessing dataset: {e}")

    # Print summary of loaded files
    print(f"Total files loaded: {len(file_list)}")
    if len(file_list) > 0:
        print(f"Sample files: {file_list[:5]}")  # Print the first 5 file paths as a sample

load_dataset(DATASET_PATH)


# In[28]:


# Set style for plots
plt.style.use("cyberpunk")

DATASET_PATH = '/kaggle/input'
TRAINING_PATH = '/kaggle/input/brain-tumor-classification-mri/Training'
TESTING_PATH = '/kaggle/input/brain-tumor-classification-mri/Testing'
IMAGE_SIZE = 150
LABELS = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

def load_images_from_folder(folder_path, labels, image_size):
    """Load and preprocess images from specified folder path."""
    X = []
    y = []
    
    for label in labels:
        label_folder_path = os.path.join(folder_path, label)
        
        # Check if folder exists
        if not os.path.exists(label_folder_path):
            print(f"Warning: {label_folder_path} does not exist.")
            continue
            
        for filename in tqdm(os.listdir(label_folder_path), desc=f"Loading {label} images"):
            img_path = os.path.join(label_folder_path, filename)
            
            # Try to read the image
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(X), np.array(y)

# Load training and testing images
X_train, y_train = load_images_from_folder(TRAINING_PATH, LABELS, IMAGE_SIZE)
X_test, y_test = load_images_from_folder(TESTING_PATH, LABELS, IMAGE_SIZE)

# Plot label distribution and sample images
def plot_label_distribution(X, y, labels):
    label_counts = {label: np.sum(y == label) for label in labels}

    plt.figure(figsize=(10, 5))
    bars = plt.bar(label_counts.keys(), label_counts.values())
    mplcyberpunk.add_bar_gradient(bars=bars)
    plt.title('Label Distribution')
    plt.show()

plot_label_distribution(X_train, y_train, LABELS)

def preprocess_data(X, y, labels, test_size=0.2, random_state=101):
    X, y = shuffle(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    y_train = tf.keras.utils.to_categorical([labels.index(label) for label in y_train])
    y_test = tf.keras.utils.to_categorical([labels.index(label) for label in y_test])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(X_train, y_train, LABELS)


# In[32]:


# Function to build and train a model
def build_and_train_model(base_model, model_name, X_train, y_train, X_test, y_test):
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TensorBoard and Checkpoints
    tensorboard_callback = TensorBoard(log_dir=f'logs/{model_name}')
    checkpoint_callback = ModelCheckpoint(filepath=f"{model_name}.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, min_delta=0.0001, verbose=1)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=1, callbacks=[tensorboard_callback, checkpoint_callback, reduce_lr_callback])
    
    return model, history

# EfficientNetB0 model
efficientnetB0 = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
model_B0, history_B0 = build_and_train_model(efficientnetB0, "EfficientNetB0", X_train, y_train, X_test, y_test)

# EfficientNetB3 model
efficientnetB3 = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
model_B3, history_B3 = build_and_train_model(efficientnetB3, "EfficientNetB3", X_train, y_train, X_test, y_test)


# In[48]:


def plot_training_results(history, model_name):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    mplcyberpunk.make_lines_glow()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name}: Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    mplcyberpunk.make_lines_glow()

    plt.tight_layout()
    plt.show()

# Plot results for both models
plot_training_results(history_B0, "EfficientNetB0")
plot_training_results(history_B3, "EfficientNetB3")


# In[59]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, model_name):
    y_true_test = np.argmax(y_test, axis=1)
    y_pred_test = np.argmax(model.predict(X_test), axis=1)

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Set white background for the plot
    sns.set(style='white')
    
    # Create heatmap with black text and white background
    sns.heatmap(confusion_matrix(y_true_test, y_pred_test), annot=True, fmt='d', cmap='Blues_r', 
                xticklabels=LABELS, yticklabels=LABELS, annot_kws={"color": "black"})
    
    # Set the titles and labels
    plt.title(f'{model_name} Confusion Matrix', color='black')
    plt.xlabel('Predicted Label', color='black')
    plt.ylabel('True Label', color='black')
    
    # Save the confusion matrix plot as an image in the Kaggle working directory
    image_path = f'/kaggle/working/{model_name}_confusion_matrix.png'
    plt.savefig(image_path, bbox_inches='tight', facecolor='white')
    
    # Display the plot
    plt.show()

    # Classification report
    print(f'{model_name} Classification Report:')
    print(classification_report(y_true_test, y_pred_test))

# Evaluate both models and save confusion matrices
evaluate_model(model_B0, X_test, y_test, "EfficientNetB0")
evaluate_model(model_B3, X_test, y_test, "EfficientNetB3")


# In[ ]:




