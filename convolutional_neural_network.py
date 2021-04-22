import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.python.client import device_lib


print(tf.__version__)
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("\n", device_lib.list_local_devices()[1])


# Preprocessing the Training set
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/path/to/train',
  seed=123,
  image_size=(64, 64),
  batch_size=32)


# Preprocessing the Test set
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/path/to/test/',
  seed=123,
  image_size=(64, 64),
  batch_size=32)

class_names = train_ds.class_names


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Data augmentation to fight overfiting
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(64, 
                                                              64,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# Create Model
num_classes = 2
cnn_model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(64, 64, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile Model
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
print(cnn_model.summary())

# Train model
from time import perf_counter
epochs=200
start_time = perf_counter()
history = cnn_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
end_time = perf_counter()
print(f'Completion Time: {end_time - start_time}')


# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Predict on new data
# img_url = "http://www.luckygoldenretriever.com/wp-content/uploads/2016/12/Cleaning_Your_Dogs_Ears.jpg"
# img_path = tf.keras.utils.get_file('dog1', origin=img_url)

# img = keras.preprocessing.image.load_img(
#     img_path, target_size=(64, 64)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) 


# predictions = cnn_model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

# save model
cnn_model.save('catdog_cnn_model.h5')


