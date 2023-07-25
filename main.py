import numpy as np
import cv2 #computer vision
import os #operating system
import random
import matplotlib.pyplot as plt
import pickle #yum

directory = r'C:\Users\Alice\Downloads\archive (1)\PetImages'
categories = ['Dog', 'Cat']

img_size = 100 #plays important role in predictions

data = []

for category in categories:
    folder = os.path.join(directory, category) #concatenate path components into path
    label = categories.index(category) #convert to binary labels
    for img in os.listdir(folder): #get list of all files & directories in specified directory
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        if img_arr is None:
            continue
        img_arr = cv2.resize(img_arr, (img_size, img_size)) #transform all images to uniform ratio
        data.append([img_arr, label])

  len(data)

#shuffle data
random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)

#convert X and y to arrays
X = np.array(X)
y = np.array(y)

len(y)

#use pickle to save X and y to computer
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))

#TRAINING
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X/255

X.shape

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))
#adding more convolutional layers
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model.fit(
    X, y, 
    epochs=25,
    validation_split=0.1,
    batch_size=64
)

#MAKING PREDICTIONS
idx2 = random.randint(0, len(y))
plt.imshow(X[idx2, :])
plt.show()

y_pred = model.predict(X[idx2, :].reshape(1, 100, 100, 3))
#print(y_pred)
y_pred = y_pred > 0.5
if (y_pred == 0):
    pred = 'dog'
else:
    pred = 'cat'
    
print('Our model says it is a: ' + pred)
