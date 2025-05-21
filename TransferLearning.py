import zipfile
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# Configuration
input_folder = "data/EuroSAT/"
train_df = "data/EuroSAT/train.csv"
val_df = "data/EuroSAT/validation.csv"
test_df = "data/EuroSAT/test.csv"
image_folder = "data/EuroSAT"
model_save_path = "WorkingLog3/"
tensorboard_logs_path = "WorkingLog3/"
figure_save_path = "WorkingLog3/"
features_labels_path = "WorkingLog3/"

batch_size = 32
img_size = (64, 64)
buffer_size = 100
learning_rate = 1e-5
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

class_labels = list(pd.read_csv(train_df)['ClassName'].unique())
class_labels = [str(label) for label in class_labels]
print(f"Class labels: {class_labels}")

from tensorflow.keras.layers import Dropout, BatchNormalization
def compile_model(input_shape, n_classes, optimizer, fine_tune=None):
    conv_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    for layer in conv_base.layers[-30:]:  # Fine-tune last 30 layers
        layer.trainable = True
    
    top_model = conv_base.output
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dropout(0.5)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs=conv_base.input, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = compile_model(input_shape, num_classes, optimizer)
model.summary()
# Model training

# Callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_save_path, 'model.weights.best.keras'),
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

# # Training parameters
N_STEPS = int(train_dataset.cardinality())  # Number of batches in training dataset
N_VAL_STEPS = int(val_dataset.cardinality())  # Number of batches in validation dataset

from tensorflow.keras.callbacks import LearningRateScheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=N_STEPS,
    epochs=epochs,
    validation_data=val_dataset,
    validation_steps=N_VAL_STEPS,
    callbacks=[early_stop, checkpoint, lr_callback]
)

# Plot the training history
def plot_history(history):
    """
    Plots the training and validation accuracy and loss over epochs.
    """
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save_path, 'training_history.png'))
    plt.show()
    

plot_history(history)

# Save the model
model.save(os.path.join(model_save_path, 'model.h5'))

# Load the best weights
model.load_weights(os.path.join(model_save_path, 'model.weights.best.keras'))
model = tf.keras.models.load_model(os.path.join(model_save_path, 'model.h5'))

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=N_VAL_STEPS)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
test_predictions = model.predict(test_dataset, steps=N_VAL_STEPS)
# Convert predictions to class labels
predicted_classes = np.argmax(test_predictions, axis=1)
# Convert true labels to class labels
true_classes = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)


# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
# Plot confusion matrix
plt.figure(figsize=(10, 8))
import seaborn as sns
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j + 0.5, i + 0.3, f'{conf_matrix_percent[i, j]:.1f}%', 
                 ha='center', va='center', color='black', fontsize=10)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(figure_save_path, 'confusion_matrix.png'))
plt.show()


# ROC AUC Curve from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

def plot_roc_auc_curve(y_true, y_probs, class_labels):
    # Binarize the true labels for multi-class ROC computation
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))
    
    # Create a figure to plot the ROC curves
    plt.figure(figsize=(10, 8))
    
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(figure_save_path, 'roc_auc_curve.png'))
    plt.show()
    

# Get the predicted probabilities for the test dataset
test_predictions_probabilities = model.predict(test_dataset)

# Plot ROC curves for each class
plot_roc_auc_curve(true_classes, test_predictions_probabilities, class_labels)

# Classification report
print(classification_report(true_classes, predicted_classes, target_names=class_labels))