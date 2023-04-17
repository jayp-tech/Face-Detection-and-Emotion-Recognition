# -*- coding: utf-8 -*-
"""

## Computer Vision COMP 6341 Project

This project captures live image, detects the face from the image and then classifies the 
emotion of the captured face
"""

"""Uploading the Dataset in Zip format and then extracting it"""
from google.colab import files
Uploaded_file = files.upload()

for fn in Uploaded_file.keys():
  print("Uploaded")

from zipfile import ZipFile
file_name = "dataset.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print("Zip Files Extracted")

"""Importing Libraries"""

import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from google.colab import files
from io import BytesIO
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""Classifying training and testing data"""

train_dir = 'train'
val_dir = 'test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


"""Defining the model"""
Model = Sequential()

Model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
Model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Flatten())

Model.add(Dense(1024, activation='relu'))
Model.add(Dropout(0.5))

Model.add(Dense(7, activation='softmax'))

Model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
Model_info = Model.fit_generator(
        train_generator,
        steps_per_epoch=455,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=90)

"""Saving the model"""
Model.save('model.h5')

"""Loading the model for future use"""
from keras.models import load_model
Emotion_Detection_Model = load_model('model.h5')

"""Defining function for emotion analysis chart"""
def emotion_analysis(emotions):
    # Define the emotions and their corresponding indices
    emotion_indices = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    
    # Get the index with the highest probability
    max_index = np.argmax(emotions)
    
    # Get the corresponding emotion label
    emotion_label = emotion_indices[max_index]
    
    # Plot a bar chart of the emotions
    fig, ax = plt.subplots()
    ax.bar(emotion_indices.values(), emotions)
    ax.set_title('Emotion Analysis')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight the emotion with the highest probability
    ax.get_children()[max_index].set_color('r')
    
    plt.show()

"""Code to capture live image """
def capture_image(filename='photo.jpg'):
    print("Press Enter to capture the image.")
    
    # Capture an image using the webcam
    from google.colab import output
    from base64 import b64decode
    from IPython.display import display, Javascript
    from google.colab.output import eval_js

    def take_photo():
        js = Javascript('''
            async function takePhoto() {
                const div = document.createElement('div');
                const capture = document.createElement('button');
                capture.textContent = 'Capture';
                div.appendChild(capture);

                const video = document.createElement('video');
                video.style.display = 'block';
                const stream = await navigator.mediaDevices.getUserMedia({video: true});

                document.body.appendChild(div);
                div.appendChild(video);
                video.srcObject = stream;
                await video.play();

                // Resize the output to fit the video element.
                google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

                // Wait for Capture to be clicked.
                await new Promise((resolve) => capture.onclick = resolve);

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                return canvas.toDataURL('image/jpeg');
            }
            ''')
        display(js)
        data = eval_js('takePhoto()')
        binary = b64decode(data.split(',')[1])
        return Image.open(BytesIO(binary))

    # Capture the image and save it as a file
    image = take_photo()
    image.save(filename, 'JPEG')
    print(f"Image captured and saved as {filename}.")

"""Capturing the image"""
filename = 'live_image.jpg'
capture_image(filename)

"""Classifying the image and fetching the results"""
def facecrop(image_file):
    facedata = '/content/haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image_file)

    try:
        faces = cascade.detectMultiScale(img)

        for (x, y, w, h) in faces:
            sub_face = img[y:y+h, x:x+w]
            cv2.imwrite('capture.jpg', sub_face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    except Exception as e:
        print (e)

    return img


if __name__ == '__main__':
    img = facecrop('/content/live_image.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48))

    x = np.array(img_resized, dtype='float32') / 255.0
    x = np.expand_dims(x, axis=0)

    custom = Emotion_Detection_Model.predict(x)
    emotion_analysis(custom[0])

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


"""Printing the confusion matrix"""
# Get the true labels and predicted labels
y_true = validation_generator.classes
y_pred = np.argmax(emotion_model.predict(validation_generator), axis=-1)
cm = confusion_matrix(y_true, y_pred)
# Print the confusion matrix
print(cm)


