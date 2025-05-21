import zipfile
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Configuration
input_folder = "data/EuroSAT/"
train_df = "data/EuroSAT/train.csv"
val_df = "data/EuroSAT/validation.csv"
test_df = "data/EuroSAT/test.csv"
image_folder = "data/EuroSAT"
model_save_path = "WorkingLog1/"
tensorboard_logs_path = "WorkingLog1/"
figure_save_path = "WorkingLog1/"

batch_size = 32
img_size = (64, 64)
buffer_size = 100
learning_rate = 0.0001
input_shape = (64, 64, 3)
num_classes = 10
epochs = 100

#Function to load the dataset
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

#Load the train test and validation datasets
train_dataset = load_dataset(train_df, image_folder, batch_size, img_size, buffer_size, True)
val_dataset = load_dataset(val_df, image_folder, batch_size, img_size, buffer_size, False)
test_dataset = load_dataset(test_df, image_folder, batch_size, img_size, buffer_size, False)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#define callbacks
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path + 'best_model.weights.h5', monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path, histogram_freq=1, write_graph=True, write_images=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
callbacks = [early_stop, checkpoint, tensorboard, lr_scheduler]

history = model.fit(train_dataset,epochs=epochs,batch_size=batch_size,validation_data=val_dataset,callbacks=callbacks)

# Plotting the training history
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(figure_save_path + 'training_history.png')
plt.show()

model.load_weights("WorkingLog1/best_model.weights.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the datasets
print("Evaluating on train dataset")
train_loss, train_acc = model.evaluate(train_dataset)
print(f"train_loss:{train_loss:.3f}:: train_accuracy: {train_acc:.3f}")

print("Evaluating on validation dataset")
val_loss, val_acc = model.evaluate(val_dataset)
print(f"val_loss:{val_loss:.3f}:: val_acc: {val_acc:.3f}")

print("Evaluating on test dataset")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"test_loss:{test_loss:.3f}:: test_acc: {test_acc:.3f}")

predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.concatenate([y.numpy() for _, y in test_dataset])

class_names = list(pd.read_csv(train_df)['ClassName'].unique())
class_names = [str(label) for label in class_names]
# class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
#                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j+0.5, i+0.6, f'\n{cm_percent[i, j]:.1f}%',
                 ha='center', va='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.savefig(figure_save_path + 'confusion_matrix.png')
plt.show()


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(true_labels)
y_test = lb.transform(true_labels)
predictions = model.predict(test_dataset)
y_pred = predictions 

# ROC AUC Curve
plt.figure(figsize=(10, 8))
for (idx, c_label) in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test[:,idx], y_pred[:,idx])
    plt.plot(fpr, tpr, label=f'{c_label} (AUC:{auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve')
plt.legend()
plt.savefig(f"{figure_save_path}roc_auc_curve.png")
plt.show()

print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred, average='macro'):.4f}")

#classification report
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels, target_names=class_names))
