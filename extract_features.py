import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
def createmodel():
    model = VGG16(weights='imagenet', include_top=False)
    model.summary()
def captureimagefeatures(path):
    video=cv2.VideoCapture(path)
    success = 1
    features=[]
    model=createmodel()
    while success:
        success,frame=video.read()
        img_data = preprocess_input(frame)
        vgg16_feature = model.predict(img_data)
        features.append(vgg16_feature)
    return features
if __name__=="__main__":
    print(captureimagefeatures(r"posit.mp4"))
