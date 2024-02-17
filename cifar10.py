# Title:        cifar10
# Author:       Ahmed Muhammed Abdelgaber
# Data:         Sat 17, 2024 
# email:         ahmedmuhammedza1998@gmail.com 
# code version: 0.0.0
#
# Copyright 2024 Ahmed Muhammed 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = \
cifar10.load_data()
X_train = X_train/255
X_test = X_test/255
## flatten the labels 


y_train, y_test = to_categorical(y_train), to_categorical(y_test)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f'shape of labels {y_train.shape} and shape shamples {X_train.shape}')
idx = 1500

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3),padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(254, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
              metrics='accuracy')
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_test, y_test))


# model.save('cifar___ahmed.h5')
plt.plot(history.history['accuracy'], label='accuracy', color='red')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy', color='black')
plt.xlabel('Epoch', color='red')
plt.ylabel('Accuracy', color='black')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
