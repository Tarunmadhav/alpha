import PIL
import numpy as np
import  pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml as foml
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression as lr
import cv2
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X=np.load('image.npz')['arr_0']
Y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(Y).value_counts())
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","Z"]
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=tts(X,Y,train_size=7500,test_size=2500,random_state=0)
xtrainscaled=xtrain/255
xtestscaled=xtest/255
clf=lr(solver="saga",multi_class="multinomial").fit(xtrainscaled,ytrain)
yprediction=clf.predict(xtestscaled)
accuracy=accuracy_score(yprediction,ytest)
print(accuracy)

cap=cv2.VideoCapture(0)
while(True):
    try:
        rit,frame=cap.read()
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=grey.shape
        upperleft=(int(width/2-56),int(height/2-56))
        bottomright=(int(width/2+56),int(height/2+56))
        cv2.rectangle(grey,upperleft,bottomright,(0,255,0),2)
        roi=grey[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert("L")
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bW_resize_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixelfilter=20
        minimumpixel=np.percentile(image_bW_resize_inverted,pixelfilter)
        imageclipped=np.clip(image_bW_resize_inverted-minimumpixel,0,255)
        maxpixel=np.max(image_bW_resize_inverted)
        imageclipped=np.asarray(imageclipped)/maxpixel
        testsample=np.array(imageclipped).reshape(1,784)
        testprediction=clf.predict(testsample)
        print("PredictedClasses",testprediction)
        cv2.imshow("frame",grey)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()