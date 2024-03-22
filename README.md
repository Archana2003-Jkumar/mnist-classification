# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/49efc8eb-075b-4f28-9662-4ce81f512ae6)

## DESIGN STEPS

### STEP 1:
Load the image dataset.
### STEP 2:
Create train and test data.
### STEP 3:
Define a model and train it.

### STEP 4:
Choose your own dataset and predict.
## PROGRAM

### Name:J.Archana Priya
### Register Number:212221230007
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[2]
single_image.shape
plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0

X_test_scaled = X_test/255.0

X_train_scaled.min()


X_train_scaled.max()

y_train[5]

y_train_onehot = utils.to_categorical(y_train,10)

y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[200]

plt.imshow(single_image,cmap='gray')
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)


model = keras.Sequential()
model.add(keras.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters =32 , kernel_size =(5,5),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))


model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=10,batch_size=52, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Archana 3007")
metrics[['accuracy','val_accuracy']].plot()

# print("Archana 3007")
metrics[['loss','val_loss']].plot

xtestpred = np.argmax(model.predict(X_test_scaled),axis=1)

print(confusion_matrix(y_test,xtestpred))

print(classification_report(y_test,xtestpred))

## <---------------  prediction for single input --------------------->

img = image.load_img('ttwo.jpg')

type(img)

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)



print(x_single_prediction)

print("Archana 3007")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/f9167840-cf65-4d2a-8eee-f4884fbb121d)

### Classification Report
![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/53eadceb-da86-47b9-b517-3cfdb3e00bfb)


### Confusion Matrix
![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/ab31244b-51f2-4123-83d1-4ee2ef330c7b)


### New Sample Data Prediction
![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/43d109a3-a890-4cb4-b8f9-c9661570491a)
![image](https://github.com/Archana2003-Jkumar/mnist-classification/assets/93427594/e6535a70-ad3e-4ff5-b35a-132532505c71)


## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images has been developed successfully.
