import cv2
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import time
from PIL import Image
import PIL.ImageOps

#loading the data using numpy and pandas
X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C' ,'D', 'E', 'F', 'G' ,'H' ,'I' ,'J' ,'K' ,'L', 'M', 'N' ,'O', 'P', 'Q','R', 'S', 'T', 'U' ,'V' ,'W' ,'X', 'Y' ,'Z']
nclasses = len(classes) #storing the length of the number of alphabets

#spliting and scaling the testing and training data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 500,random_state = 9,train_size=3500) 
x_scale = x_train/255.0
x_test_scale = x_test/255.0

#creating a Logistic Regression model and fitting our data
clf = LogisticRegression(solver = 'saga', multi_class='multinomial')
clf.fit(x_scale,y_train)

def getPrediction (i):
    opened_image = Image.open(i)
    gray = opened_image.convert('L')
    resized_image = gray.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    minimum_pixel = np.percentile(resized_image,pixel_filter)
    scaled_image = np.clip(resized_image-minimum_pixel,0,255)
    maximumpixel = np.max(resized_image)
    scaled_image = np.asarray(scaled_image)/maximumpixel
    test = np.array(scaled_image).reshape(1,784)
    prediction = clf.predict(test)
    return prediction[0]
