#!/usr/bin/env python
# coding: utf-8

# # Building a CNN for image classification 

# In[2]:


import tensorflow as tf
import numpy as np

# Load the CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data to [0, 1]
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Combine training and test data for easy splitting
x_all = np.concatenate([x_train_full, x_test], axis=0)
y_all = np.concatenate([y_train_full, y_test], axis=0)

# Calculate sizes for each split (70% training, 15% validation, 15% test)
num_samples = x_all.shape[0]
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

# Shuffle the dataset
indices = np.random.permutation(num_samples)
x_all, y_all = x_all[indices], y_all[indices]

# Create the training, validation, and test sets
x_train, y_train = x_all[:train_size], y_all[:train_size]
x_val, y_val = x_all[train_size:train_size+val_size], y_all[train_size:train_size+val_size]
x_test, y_test = x_all[train_size+val_size:], y_all[train_size+val_size:]

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.1),       # Randomly rotate images by up to 10%
    tf.keras.layers.RandomZoom(0.1),           # Randomly zoom images by up to 10%
    tf.keras.layers.RandomContrast(0.1)        # Randomly adjust contrast
])

# Define a function to apply augmentation to batches
def augment_data(x, y):
    return data_augmentation(x, training=True), y

# Create a dataset object for efficient augmentation and batching
batch_size = 32

# Training dataset with augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).map(augment_data)

# Validation and test datasets without augmentation
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Example: Preview augmented images
for x_batch, y_batch in train_dataset.take(1):
    print(f"Augmented batch shape: {x_batch.shape}, Labels batch shape: {y_batch.shape}")


# In[3]:


import tensorflow as tf

# Define the CNN model
model = tf.keras.Sequential()

# Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# MaxPooling Layer 1: 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer 1: Dropout rate 0.25
model.add(tf.keras.layers.Dropout(0.25))

# Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# MaxPooling Layer 2: 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer 2: Dropout rate 0.25
model.add(tf.keras.layers.Dropout(0.25))

# Conv Layer 3: 128 filters, 3x3 kernel, ReLU activation
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# MaxPooling Layer 3: 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer 3: Dropout rate 0.25
model.add(tf.keras.layers.Dropout(0.25))

# Flatten the feature map to feed into fully connected layers
model.add(tf.keras.layers.Flatten())

# Fully Connected Layer 1: 128 units, ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Dropout Layer 4: Dropout rate 0.5
model.add(tf.keras.layers.Dropout(0.5))

# Output Layer: 10 units (for CIFAR-10, which has 10 classes), softmax activation
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


# In[4]:


# Compile the model with categorical crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Use categorical crossentropy for one-hot encoded labels
              metrics=['accuracy'])            # Metric: Accuracy


# In[5]:


# Compile the model with sparse categorical crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # Use categorical crossentropy for one-hot encoded labels
              metrics=['accuracy'])            # Metric: Accuracy


# In[8]:


import tensorflow as tf

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',            # Metric to monitor
    patience=3,                    # Number of epochs with no improvement to wait
    restore_best_weights=True      # Restore model weights from the epoch with the best value of the monitored metric
)

# Define model checkpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',            # Path to save the model (with .keras extension)
    monitor='val_loss',            # Metric to monitor
    save_best_only=True,            # Save only the best model
    mode='min',                    # Mode for monitoring 'val_loss' (minimizing)
    verbose=1                      # Verbosity mode (1: progress bar)
)

# Train the model with callbacks
history = model.fit(
    train_dataset,                 # Training data (with augmentation)
    epochs=10,                     # Number of epochs to train
    validation_data=val_dataset,  # Validation data
    batch_size=32,                # Batch size
    callbacks=[early_stopping, model_checkpoint],  # List of callbacks
    verbose=1                     # Verbosity mode (1: progress bar)
)


# In[11]:


import matplotlib.pyplot as plt

#plotting of training and validation accuracy values
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[12]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Load the saved model
model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict labels for the test set
y_test_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_test_pred_prob = model.predict(test_dataset)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Convert one-hot encoded test labels to integer labels
y_test_true_labels = np.argmax(y_test_true, axis=1)

# Prepare classification report
report = classification_report(y_test_true_labels, y_test_pred, target_names=[f'Class {i}' for i in range(10)])

print("Classification Report:")
print(report)

# Compute accuracy from the test set predictions
accuracy = accuracy_score(y_test_true_labels, y_test_pred)
print(f"Accuracy: {accuracy:.4f}")


# In[13]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the saved model
model = tf.keras.models.load_model('best_model.keras')

# Predict labels for the test set
y_test_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_test_pred_prob = model.predict(test_dataset)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Convert one-hot encoded test labels to integer labels
y_test_true_labels = np.argmax(y_test_true, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_true_labels, y_test_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(10)],
            yticklabels=[f'Class {i}' for i in range(10)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('best_model.keras')

# Select a few images from the test set
def get_sample_images(test_dataset, num_samples=5):
    images = []
    labels = []
    for img_batch, lbl_batch in test_dataset.take(1):
        # Flatten the dataset to get images and labels
        images = img_batch[:num_samples]
        labels = lbl_batch[:num_samples]
    return images, labels

num_samples = 5
sample_images, sample_labels = get_sample_images(test_dataset, num_samples=num_samples)

# Predict labels for the sample images
sample_preds_prob = model.predict(sample_images)
sample_preds = np.argmax(sample_preds_prob, axis=1)
sample_labels = np.argmax(sample_labels, axis=1)

# Plot the sample images with their true and predicted labels
plt.figure(figsize=(15, 10))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i].numpy().astype(np.uint8))
    plt.title(f'True: {sample_labels[i]}\nPred: {sample_preds[i]}')
    plt.axis('off')
plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('best_model.keras')

# Select a few images from the test set
def get_sample_images(test_dataset, num_samples=5):
    images = []
    labels = []
    for img_batch, lbl_batch in test_dataset.take(1):
        images = img_batch[:num_samples]
        labels = lbl_batch[:num_samples]
    return images, labels

num_samples = 5
sample_images, sample_labels = get_sample_images(test_dataset, num_samples=num_samples)

# Predict labels for the sample images
sample_preds_prob = model.predict(sample_images)
sample_preds = np.argmax(sample_preds_prob, axis=1)
sample_labels = np.argmax(sample_labels, axis=1)

# Plot the sample images with their true and predicted labels
plt.figure(figsize=(15, 10))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i].numpy())
    plt.title(f'True: {sample_labels[i]}\nPred: {sample_preds[i]}')
    plt.axis('off')
plt.show()


# In[ ]:




