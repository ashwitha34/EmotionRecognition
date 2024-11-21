# -*- coding: utf-8 -*-
"""emotionrgn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13xpcY8Fn1E4nYheL9ub1UP3ZmiStKPPD
"""

!mkdir -p ~/.kaggle

!cp /content/kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

_author_ = "madhaviii"

pwd

cd /root

mkdir .kaggle

cd /root/.kaggle

!cp /content/kaggle.json ~/.kaggle

pwd

ls

ls

!pip install --upgrade --force-reinstall --no-deps kaggle

!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge

ls

!unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip

ls

!tar -xf fer2013.tar.gz

ls

cd fer2013

ls

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
from matplotlib.colors import ListedColormap
import seaborn as sns

df = pd.read_csv("/root/.kaggle/fer2013/fer2013.csv")
df.head()

df

df.shape

plt.figure(figsize=(9,4))
sns.countplot(x='emotion', data=df)

df['emotion'].value_counts()

plt.figure(figsize=(9,4))
sns.countplot(x='Usage', data=df)

df['Usage'].value_counts()

import cv2
image_size=(48,48)

pixels = df['pixels'].tolist() # Converting the relevant column element into a list for each row
width, height = 48, 48
faces = []

for pixel_sequence in pixels:
  face = [int(pixel) for pixel in pixel_sequence.split(' ')] # Splitting the string by space character as a list
  face = np.asarray(face).reshape(width, height) #converting the list to numpy array in size of 48*48
  face = cv2.resize(face.astype('uint8'),image_size) #resize the image to have 48 cols (width) and 48 rows (height)
  faces.append(face.astype('float32')) #makes the list of each images of 48*48 and their pixels in numpyarray form

faces = np.asarray(faces) #converting the list into numpy array
faces = np.expand_dims(faces, -1) #Expand the shape of an array -1=last dimension => means color space
emotions = pd.get_dummies(df['emotion']).to_numpy() #doing the one hot encoding type on emotions

print(faces[0])

print(faces.shape)
print(faces[0].ndim)
print(type(faces))

print(emotions[0])

print(emotions.shape)
print(emotions.ndim)
print(type(emotions))

x = faces.astype('float32')
x = x / 255.0 #Dividing the pixels by 255 for normalization  => range(0,1)
# Scaling the pixels value in range(-1,1)
x = x - 0.5
x = x * 2.0

print(x[0])

type(x)

plt.plot(x[0,0])
plt.show()

print(x.min(),x.max())

num_samples, num_classes = emotions.shape
num_samples = len(x)
num_train_samples = int((1 - 0.2)*num_samples)
# Traning data
train_x = x[:num_train_samples]
train_y = emotions[:num_train_samples]
# Validation data
val_x = x[num_train_samples:]
val_y = emotions[num_train_samples:]
train_data = (train_x, train_y)
val_data = (val_x, val_y)

print('Training Pixels',train_x.shape)  # ==> 4 dims -  no of images , width , height , color
print('Training labels',train_y.shape)
print('Validation Pixels',val_x.shape)
print('Validation labels',val_y.shape)

from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

"""
* keras.__version__
* pip install --upgrade keras
"""

input_shape=(48, 48, 1)
num_classes = 7

""" Building up Model Architecture """

model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))

model.add(BatchNormalization())
model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax',name='predictions'))

model.summary()

# parameters
batch_size = 32 #Number of samples per gradient update
num_epochs = 200 # Number of epochs to train the model.
#input_shape = (64, 64, 1)
verbose = 1 #per epohs  progress bar
num_classes = 7
patience = 50
base_path = 'drive/Colab Notebooks/emotion/simplecnn/'

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data generator Generate batches of tensor image data with real-time data augmentation
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

datasets = ['fer2013']
num_epochs = 30
base_path="/content"
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    #callbacks
    log_file_path = dataset_name + '_emotion_training.log'

    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)

    trained_models_path = base_path + dataset_name + 'simple_cnn'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.keras'  # Changed .hdf5 to .keras
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    my_callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


    # loading dataset
    train_faces, train_emotions = train_data
    history = model.fit(data_generator.flow(train_faces, train_emotions, batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=my_callbacks, validation_data=val_data)
       #not callbacks = [my_callbacks] since we my_callbacks is already a list

#evaluate() returns [loss,acc]
score = model.evaluate(val_x, val_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

history_dict=history.history
history_dict.keys()

print(history_dict["accuracy"])

import matplotlib.pyplot as plt

train_loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, train_loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

emotion_dict = {0: "Neutral", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
emojis = {
    0: "\U0001f620",  # Angry face
    1: "\U0001f922",  # Nauseated face
    2: "\U0001f628",  # Fearful face
    3: "\U0001f60A",  # Smiling face
    4: "\U0001f625",  # Sad face
    5: "\U0001f632",  # Shocked face
    6: "\U0001f610"   # Neutral face
}

print(emojis.values(),sep=" ")

!cd content
!pwd
!ls

from google.colab.patches import cv2_imshow
import cv2

def _predict(path):
  facecasc = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
  imagePath = '/content/'+path
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=10)
  print("No of faces : ",len(faces))
  i = 1
  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]                      #croping
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)

    maxindex = int(np.argmax(prediction))
    print("person ",i," : ",emotion_dict[maxindex], "-->",emojis[maxindex])
    cv2.putText(image, emotion_dict[maxindex], (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                              #if text is not apeared , change coordinates. it may work

  cv2_imshow(image)

from google.colab.patches import cv2_imshow
import cv2

def _predict(path):
  facecasc = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
  imagePath = '/content/'+path
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=10)
  print("No of faces : ",len(faces))
  i = 1
  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]                      #croping
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)

    maxindex = int(np.argmax(prediction))
    print("person ",i," : ",emotion_dict[maxindex], "-->",emojis[maxindex])
    cv2.putText(image, emotion_dict[maxindex], (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                              #if text is not apeared , change coordinates. it may work

  cv2_imshow(image)

pip install requests

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Load the Haar Cascade face detector
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def _predict(url):
    # Download the image from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")
        return

    # Open the image
    pil_image = Image.open(BytesIO(response.content))

    # Convert it to RGB format to ensure compatibility
    pil_image = pil_image.convert("RGB")

    # Convert the PIL image to a NumPy array (OpenCV format)
    image = np.array(pil_image)

    # Convert the RGB image to BGR (OpenCV's color format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade face detector
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region
        face = gray[y:y+h, x:x+w]

        # Resize to the required input size of your model (e.g., 48x48 for FER-2013)
        face_resized = cv2.resize(face, (48, 48))

        # Normalize pixel values between 0 and 1
        face_resized = face_resized.astype("float32") / 255.0

        # Expand dimensions to fit the model input shape (e.g., (1, 48, 48, 1))
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Predict emotion using your model (assuming 'model' is already loaded)
        emotion_prediction = model.predict(face_resized)

        # Get the predicted emotion class
        emotion_label_arg = np.argmax(emotion_prediction)

        # Dictionary mapping indices to emotion labels
        emojis = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        # Get the corresponding emotion label
        emotion_label = emojis.get(emotion_label_arg, "Unknown emotion")

        # Put the emotion label above the bounding box
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert back to RGB (for display in matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes and emotion labels
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

# Call the function with the URL to the image
_predict("https://thumbs.dreamstime.com/b/portrait-attractive-cheerful-young-man-smiling-happy-face-human-expressions-emotions-latin-model-beautiful-smile-150784563.jpg")

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Load the Haar Cascade face detector
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def _predict(url):
    # Download the image from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")
        return

    # Open the image
    pil_image = Image.open(BytesIO(response.content))

    # Convert it to RGB format to ensure compatibility
    pil_image = pil_image.convert("RGB")

    # Convert the PIL image to a NumPy array (OpenCV format)
    image = np.array(pil_image)

    # Convert the RGB image to BGR (OpenCV's color format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade face detector
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region
        face = gray[y:y+h, x:x+w]

        # Resize to the required input size of your model (e.g., 48x48 for FER-2013)
        face_resized = cv2.resize(face, (48, 48))

        # Normalize pixel values between 0 and 1
        face_resized = face_resized.astype("float32") / 255.0

        # Expand dimensions to fit the model input shape (e.g., (1, 48, 48, 1))
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Predict emotion using your model (assuming 'model' is already loaded)
        emotion_prediction = model.predict(face_resized)

        # Get the predicted emotion class
        emotion_label_arg = np.argmax(emotion_prediction)

        # Dictionary mapping indices to emotion labels
        emojis = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        # Get the corresponding emotion label
        emotion_label = emojis.get(emotion_label_arg, "Unknown emotion")

        # Put the emotion label above the bounding box
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert back to RGB (for display in matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes and emotion labels
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

# Call the function with the URL to the image
_predict("https://thumbs.dreamstime.com/b/human-expressions-emotions-young-attractive-man-sad-face-looking-depressed-unhappy-close-up-portrait-handsome-crying-153222343.jpg")

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Load the Haar Cascade face detector
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def _predict(url):
    # Download the image from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")
        return

    # Open the image
    pil_image = Image.open(BytesIO(response.content))

    # Convert it to RGB format to ensure compatibility
    pil_image = pil_image.convert("RGB")

    # Convert the PIL image to a NumPy array (OpenCV format)
    image = np.array(pil_image)

    # Convert the RGB image to BGR (OpenCV's color format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade face detector
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region
        face = gray[y:y+h, x:x+w]

        # Resize to the required input size of your model (e.g., 48x48 for FER-2013)
        face_resized = cv2.resize(face, (48, 48))

        # Normalize pixel values between 0 and 1
        face_resized = face_resized.astype("float32") / 255.0

        # Expand dimensions to fit the model input shape (e.g., (1, 48, 48, 1))
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Predict emotion using your model (assuming 'model' is already loaded)
        emotion_prediction = model.predict(face_resized)

        # Get the predicted emotion class
        emotion_label_arg = np.argmax(emotion_prediction)

        # Dictionary mapping indices to emotion labels
        emojis = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        # Get the corresponding emotion label
        emotion_label = emojis.get(emotion_label_arg, "Unknown emotion")

        # Put the emotion label above the bounding box
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert back to RGB (for display in matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes and emotion labels
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

# Call the function with the URL to the image
_predict("https://th.bing.com/th/id/OIP.Gv4ZLWYc96Be-wyGGoV7uAHaFX?w=256&h=185&c=7&r=0&o=5&pid=1.7")