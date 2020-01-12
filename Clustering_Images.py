from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image


def data_preprocessing(folder):
    """Takes in the location of the folder containing the pictures and loads all the pictures from the folder
    for pre-processing. Data pre-processing includes converting the image to gray scale and resizing the image
    to 50 X 50 pixels."""
    images = []
    num_images = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Converts the image to gray scale
            new_img = np.array(Image.fromarray(img).resize((50, 50), Image.ANTIALIAS))  # Resize the images to 50 X 50
            images.append(new_img)
            num_images += 1
    return images, num_images


faces, num_i = data_preprocessing("DetectedFaces/Faces")


image_list = []
for i in range(num_i):
    # Converts the multidimensional array which contains the pixel values into single dimension and stores it in form
    # of a pandas series in a list
    image_list.append(pd.Series(faces[i].ravel()))

image_list = StandardScaler().fit_transform(image_list)  # Pre-processing data step 2
image_data_frame = pd.DataFrame(image_list)  # Creates a pandas data frame and each row represents a single image and
# each column represents a pixel value.
image_data_frame.columns = ['Pixels']*2500
print(image_data_frame)


# Use Principal Component Analysis to reduce the 2500 pixels to 2 pixels
pca = PCA(n_components=2)
red_data_frame = pca.fit_transform(image_data_frame)
red_data_frame = pd.DataFrame(red_data_frame)
red_data_frame.columns = ['Red_Pixels1', 'Red_Pixels2']
print(red_data_frame)


# # Adjust the values of epsilon and min_samples
# f = open('Clusters/X1', 'w')
# epsilon = 10
# step = 0.5
# for i in range(1, 10000):
#     db = DBSCAN(eps=epsilon, min_samples=10).fit(image_data_frame)
#     labels1 = db.labels_
#     f.write(str(labels1))
#     f.write("\n")
#     epsilon += step


# Using the pre calculated eps and min_samples in the DBSCAN algorithm
db = DBSCAN(eps=70, min_samples=30).fit(image_data_frame)
labels1 = db.labels_
print(labels1)



