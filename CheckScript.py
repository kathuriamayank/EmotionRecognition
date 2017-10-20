import numpy as np
import matplotlib.pyplot as plt
import cv2

#Using The Training xml File
fisherface=cv2.createFisherFaceRecognizer()
fisherface.load("trainedmodel.xml")


#Prediction
#Enter The Image Path
path="sonali_happy1.jpg"
img=cv2.imread(path)
img_gray=np.asarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
img_gray=cv2.resize(img_gray,(268,268))
pred,conf=fisherface.predict(img_gray)
print pred,conf

if pred==0:
    print "Angry"
if pred==1:
    print "Happy"
if pred==2:
    print "Sad"
if pred==3:
    print "Surprise"
print "The confidence is ",conf




