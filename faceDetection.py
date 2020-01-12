import cv2
import numpy as np
import os
import random


def face_detection(image):
    """Takes in the ndarray of image , detects the faces and stores the extracted faces in one folder and the outlined
    picture of the face in another folder"""
    prof_pic = np.copy(image)
    gray_pic = cv2.cvtColor(prof_pic, cv2.COLOR_RGB2GRAY)  # Converts the picture to gray-scale
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray_pic,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print("Found {0} Faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(prof_pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite("DetectedFaces/Faces/x{}.jpg".format(random.randint(1, 10000)), roi_color)
    status = cv2.imwrite('DetectedFaces/Outline/faces_detected{}.jpg'.format(random.randint(1, 10000)), prof_pic)
    print("Image faces_detected.jpg written to filesystem: ", status)
    return None


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


sample = load_images_from_folder('FaceDProject')
for i in sample:
    face_detection(i)

