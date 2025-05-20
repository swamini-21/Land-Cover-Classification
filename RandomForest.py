import zipfile
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Configuration
input_folder = "data/EuroSAT/"
train_df = "data/EuroSAT/train.csv"
val_df = "data/EuroSAT/validation.csv"
test_df = "data/EuroSAT/test.csv"
image_folder = "data/EuroSAT"
model_save_path = "WorkingLog2/"
tensorboard_logs_path = "WorkingLog2/"
figure_save_path = "WorkingLog2/"
features_labels_path = "WorkingLog2/"

batch_size = 32
img_size = (64, 64)
buffer_size = 100
learning_rate = 0.0001
input_shape = (64, 64, 3)
num_classes = 10
epochs = 100

def load_dataset(csv_path, image_folder, batch_size, img_size, buffer_size, shuffle):
    df = pd.read_csv(csv_path)
    image_paths = [f"{image_folder}/{filename}" for filename in df.Filename]
    labels = df.Label.values

    def load_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        
        # Enhanced data augmentations for better generalization
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # New augmentation for saturation
        image = tf.image.random_hue(image, max_delta=0.1)  # New augmentation for hue
        angle = np.random.uniform(0, 360)  # Random angle between 0 and 360 degrees
        image = tf.image.rot90(image, k=int(angle // 90))  # Rotate in 90-degree increments
        
        return tf.cast(image, tf.float32) / 255.0, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    return dataset.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

# Load train, validation, and test datasets using the new function
train_dataset = load_dataset(train_df, image_folder, batch_size, img_size, buffer_size, shuffle=True)
val_dataset = load_dataset(val_df, image_folder, batch_size, img_size, buffer_size, shuffle=False)
test_dataset = load_dataset(test_df, image_folder, batch_size, img_size, buffer_size, shuffle=False)

# Feature extraction
def extract_features(dataset):
    features = []
    labels = []
    for image_batch, label_batch in dataset:
        # Flatten the image data to use as features
        flattened_images = image_batch.numpy().reshape(image_batch.shape[0], -1)  # Flatten to 1D array
        features.append(flattened_images)
        labels.append(label_batch)
    
    return np.vstack(features), np.concatenate(labels)

# Extract features for train, validation, and test sets
X_train_features, y_train_labels = extract_features(train_dataset)
X_val_features, y_val_labels = extract_features(val_dataset)
X_test_features, y_test_labels = extract_features(test_dataset)

# Print dataset sizes
print(f"Train Features: {X_train_features.shape}, Validation Features: {X_val_features.shape}, Test Features: {X_test_features.shape}")

# Label encoding
label_encoder = LabelEncoder()
print("Label Encoding Started")
y_train = label_encoder.fit_transform(y_train_labels)
y_test = label_encoder.transform(y_test_labels)
y_val = label_encoder.transform(y_val_labels)
print("Label Encoding Completed")

from sklearn.preprocessing import StandardScaler
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_val_scaled = scaler.transform(X_val_features)
X_test_scaled = scaler.transform(X_test_features)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
from sklearn.ensemble import RandomForestClassifier

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train_scaled, y_train)
print("Random Forest Classifier Training Completed.")

# Save the trained model
import joblib
joblib.dump(rf_classifier, model_save_path + "model.joblib")
print(f"Model saved to {model_save_path}")

# Evaluate the model
y_train_pred = rf_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_val_pred = rf_classifier.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

y_test_pred = rf_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

train_accuracies = [train_accuracy]  # In case you want to plot at multiple steps
val_accuracies = [val_accuracy]
test_accuracies = [test_accuracy]

class_names = list(pd.read_csv(train_df)['ClassName'].unique())
class_names = [str(label) for label in class_names]

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j+0.5, i+0.6, f'\n{cm_percent[i, j]:.1f}%', ha='center', va='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.savefig(figure_save_path + 'confusion_matrix.png')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# Get predicted probabilities for each class
y_test_pred_probabilities = rf_classifier.predict_proba(X_test_scaled)
y_test_binarized = label_binarize(y_test, classes=list(range(num_classes)))

# Plot ROC curves
plt.figure(figsize=(10, 8))
for (idx, c_label) in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, idx].astype(int), y_test_pred_probabilities[:, idx])
    plt.plot(fpr, tpr, label=f'{c_label} (AUC:{auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend()
plt.savefig(f"{figure_save_path}roc_auc_curve.png")
plt.show()

print(f"ROC AUC Score (macro average): {roc_auc_score(y_test_binarized, y_test_pred_probabilities, average='macro'):.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))
# Save the classification report to a text file
with open(os.path.join(figure_save_path, 'classification_report.txt'), 'w') as f:
    f.write(classification_report(y_test, y_test_pred, target_names=class_names))
