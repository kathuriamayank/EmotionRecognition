
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

img=cv2.imread("image.jpg")

#plt.imshow(img,cmap="gray")
#plt.show()
cap=cv2.VideoCapture(0)


# In[4]:


fisherface=cv2.createFisherFaceRecognizer()
fisherface.load("trainedwithoutsad.xml")
while 1:
    cv2.namedWindow("RawWindow")
    cv2.namedWindow("FaceSwappedWindow")
    ret,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img_gray)
    for (x,y,w,h) in faces:
        #img,initial points,width and height of rect,color of rect,thichkness
        
        cv2.imshow("RawWindow",img_gray)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=img_gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(268,268))

        
       
        

        '''img=cv2.imread("Emoji/happy.png")
        gray=np.array(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        gray=cv2.resize(gray,(h,w))
        img_gray[y:y+h,x:x+w]=gray
        cv2.imshow("FaceSwappedWindow",img_gray)'''
    
    #cv2.imshow("image",img)
    key=cv2.waitKey(30) & 0xff
    if key==27:    #esc key
        
        break
    if key==ord("s"):
        cv2.imwrite("face.jpg",roi_gray)
        face=cv2.imread("face.jpg")
        face_gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face_gray=cv2.resize(face_gray,(268,268))
        pred,conf=fisherface.predict(face_gray)
        if pred==0:
            print "Angry"
            emoji=cv2.imread("Emoji/anger.png")
            emoji_gray=np.array(cv2.cvtColor(emoji,cv2.COLOR_BGR2GRAY))
            emoji_gray=cv2.resize(emoji_gray,(h,w))
            img_gray[y:y+h,x:x+w]=emoji_gray
            cv2.imshow("FaceSwappedWindow",img_gray)
        if pred==1:
            print "Happy"
            emoji=cv2.imread("Emoji/happy.png")
            emoji_gray=np.array(cv2.cvtColor(emoji,cv2.COLOR_BGR2GRAY))
            emoji_gray=cv2.resize(emoji_gray,(h,w))
            img_gray[y:y+h,x:x+w]=emoji_gray
            cv2.imshow("FaceSwappedWindow",img_gray)
            

        if pred==3:
            print "Surprise"
            emoji=cv2.imread("Emoji/surprise.png")
            emoji_gray=np.array(cv2.cvtColor(emoji,cv2.COLOR_BGR2GRAY))
            emoji_gray=cv2.resize(emoji_gray,(h,w))
            img_gray[y:y+h,x:x+w]=emoji_gray
            cv2.imshow("FaceSwappedWindow",img_gray)
        
        
        
        
        

cap.release()
cv2.destroyAllWindows()
    


# In[ ]:



