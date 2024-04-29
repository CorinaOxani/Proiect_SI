import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split  # Corrected import
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  # Corrected MaxPooling2D

data = []
labels = []
classes = 43
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))  # Corrected path joining
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))  # Corrected path joining
            image = image.resize((30, 30))  # Uniform image size for input
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image {a}")

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
# Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)
# Converting the labels into one hot encoding
y_t1 = to_categorical(y_t1, classes)
y_t2 = to_categorical(y_t2, classes)


# Definirea modelului
model = Sequential()

# Adăugarea stratului Conv2D cu 32 de filtre și kernel de dimensiunea (5,5)
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# Adăugarea unui strat de MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adăugarea unui strat Dropout pentru a reduce overfitting-ul
model.add(Dropout(rate=0.25))

# Alte două straturi Conv2D, acum cu 64 de filtre
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# MaxPooling și Dropout
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Flatten outputul pentru a putea fi introdus în straturile Dense
model.add(Flatten())
# Primul strat Dense complet conectat
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# Stratul Dense de output cu activare softmax pentru clasificare multi-clasă
model.add(Dense(43, activation='softmax'))

# Compilarea modelului
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Antrenarea modelului
epochs = 15
batch_size = 32
history = model.fit(X_t1, y_t1, batch_size=batch_size, epochs=epochs, validation_data=(X_t2, y_t2))


#plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
   image = Image.open(img)
   image = image.resize((30,30))
   data.append(np.array(image))
X_test = np.array(data)
pred = model.predict(X_test)  # Use predict instead of predict_classes
pred_classes = np.argmax(pred, axis=1)  # Convert probabilities to class labels
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred_classes)) 

model.save('traffic_classifier.h5')#to save