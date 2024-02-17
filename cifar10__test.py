import numpy as np 
from keras.datasets import cifar10
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
_ , (X_test, y_test) = \
cifar10.load_data()


labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_test =  to_categorical(y_test)



model = tf.keras.models.load_model('cifar___ahmed.h5')
prediction = model.predict(X_test[0].reshape(-1, 32, 32, 3))
print(y_test[0])
indx = np.where(prediction[0]==1.0)[0][0]
print(labels[indx])
plt.imshow(X_test[0])
plt.title("Test cifar10")
props = dict(boxstyle='round', facecolor='black', alpha=0.5)
plt.text(0.5 , 2, labels[indx], fontsize=16, fontfamily='cursive',c='yellow', bbox=props)
plt.show()
# model.summary()
