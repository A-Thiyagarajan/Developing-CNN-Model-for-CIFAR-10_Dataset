# Developing CNN Model for CIFAR-10 Dataset
## Program:
```
# Name:Thiyagarajan A
# Reg.no: 212222240110 
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import datasets, layers, models
from tensorflow.keras import utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
```
```
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```
```
# Checking the number of rows (records) and columns (features)
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
```
![w1](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/7fc75a49-aab4-4837-8310-15322f1d7e27)



```
# Checking the number of unique classes 
print(np.unique(train_labels))
print(np.unique(test_labels))
```

![w2](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/c67d1c1a-b484-4255-9368-ca8fb683858a)



```
# Creating a list of all the class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```


```
# Visualizing some of the images from the training dataset
plt.figure(figsize=[10,10])
for i in range (25):    # for first 25 images
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```


![w3](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/8ca91ee6-28c9-48a4-9512-1de735a6c544)



```
# Converting the pixels data to float type
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
 
# Standardizing (255 is the total number of pixels an image can have)
train_images = train_images / 255
test_images = test_images / 255 

# One hot encoding the target class (labels)
num_classes = 10
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)
```


```# Creating a sequential model and adding layers to it

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

# Checking the model summary
model.summary()
```

![w4](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/1be6c38f-2ce8-4dad-b495-ecdf723de974)




```
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
```

```
history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_data=(test_images, test_labels))
```


![w5](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/56cdbdfa-7581-4aae-b85b-ddd10cc08e5b)



```
# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
```

![loss-c](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/01a55d1d-a684-434c-aa60-e7d3eecc80b3)



```
# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
```


![acc](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/bb9f448e-ab1d-4b56-8b27-e8767a00009f)



```
# Making the Predictions
pred = model.predict(test_images)
print(pred)

# Converting the predictions into label index 
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)
```

![p](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/91240e27-663c-47f5-8844-6c028070dcab)


```
# Plotting the Actual vs. Predicted results

fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(test_images[i])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[np.argmax(test_labels[i])], class_names[pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
```





![w7](https://github.com/A-Thiyagarajan/Developing-CNN-Model-for-CIFAR-10_Dataset/assets/118707693/d419e6be-78bd-419b-96bb-15081d78c4cd)


































