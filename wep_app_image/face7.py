import streamlit as st
import cv2,time
from PIL import Image,ImageEnhance
from matplotlib import pyplot as plt
import numpy
import numpy as np
import os
import json
import streamlit.components.v1 as components
import base64
from io import BytesIO

try:
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    

except Exception:
    st.write("Error loading cascade classifiers")



def detect(image):
    image=np.array(image.convert('RGB'))
    img=cv2.cvtColor(image,1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image=image,scaleFactor=1.3,minNeighbors=5)
    #faces=face_cascade.detectMultiScale(image=image,1.3,5)
    for (x,y,w,h) in faces:
        img=cv2.rectangle(img=image,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,0),thickness=4)
        #blur=img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
        
        
        
        
    return image,faces

def blur(image):
    image=np.array(image.convert('RGB'))
    img=cv2.cvtColor(image,1)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image=image,scaleFactor=1.3,minNeighbors=5)
    #faces=face_cascade.detectMultiScale(image=image,1.3,5)
    for (x,y,w,h) in faces:
        img=cv2.rectangle(img=image,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,0),thickness=4)
        blur=img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
        
        
        
        #roi=image[y:y+h,x:x+w]
        #eyes=eye_cascade.detectMultiScale(roi,scaleFactor=1.5,minNeighbors=5)
        #smile=smile_cascade.detectMultiScale(roi,scaleFactor=1.5,minNeighbors=25)
        #for(ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(255,0,0),3)
        
        #for(sx,sy,sw,sh) in smile:
            #cv2.rectangle(roi,(sx,sy),(sx+sw,sy+sh),(255,0,0),3)
    return image,faces

def hsv(image1):
    image1=np.array(image1.convert('RGB'))
    hsv_image=cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
    return image1,hsv_image


def about():
    
    st.write('''
    **Haar cascade is an object detection algorithm
    Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle**.''')
    img=Image.open("haar_features.jpg")
    st.image(img,width=300,caption="About Haar Cascade Algorithm")
    st.write('''**Haar Features
Now all possible sizes and locations of each kernel is used to calculate plenty of features. (Just imagine how much computation it needs? Even a 24x24 window results over 160000 features). For each feature calculation, we need to find sum of pixels under white and black rectangles. To solve this, they introduced the integral images. It simplifies calculation of sum of pixels, how large may be the number of pixels, to an operation involving just four pixels. Nice, isn’t it? It makes things super-fast.

But among all these features we calculated, most of them are irrelevant. For example, consider the image below. Top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applying on cheeks or any other place is irrelevant. So how do we select the best features out of 160000+ features? It is achieved by Adaboost.

Face Detection
For this, we apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. But obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that best classifies the face and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then again same process is done. New error rates are calculated. Also new weights. The process is continued until required accuracy or error rate is achieved or required number of features are found).

Final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can’t classify the image, but together with others forms a strong classifier. The paper says even 200 features provide detection with 95% accuracy. Their final setup had around 6000 features. (Imagine a reduction from 160000+ features to 6000 features. That is a big gain)**.


    ''')
    img=Image.open("haar.png")
    st.image(img,width=300,caption="About Haar Cascade Algorithm")

def cannys(my_image):
    my_image=np.array(my_image)
    img_gray=cv2.cvtColor(my_image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(my_image,100,200)
    return my_image,edges


def mask(our_image):
    #our_image=np.array(our_image.convert('RGB'))
    our_image=np.array(our_image)
    our_image=our_image[:,:,::-1].copy()
    img_gray=cv2.cvtColor(our_image,cv2.COLOR_BGR2GRAY)
    img_invert=cv2.bitwise_not(img_gray)
    blur=cv2.GaussianBlur(img_invert,(21,21),0)
    invertedblur=cv2.bitwise_not(blur)
    sketch=cv2.divide(img_gray,invertedblur,scale=256.0)
    
    #p=cv2.imshow("image",sketch)
    
    return our_image,sketch

def cartoon(t_image):
    t_image=np.array(t_image)
    #t_image=t_image[:,:,::-1].copy()
    grey=cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
    greys=cv2.medianBlur(grey,5)
    edges=cv2.adaptiveThreshold(greys,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color=cv2.bilateralFilter(t_image,9,250,250)
    cartoos=cv2.bitwise_and(color,color,mask=edges)
    #st.image(cartoos)
    return t_image,cartoos

def cartoon_vid():
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture(0)
    first_frame=None
    
    while True:
        check,frame=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        greys=cv2.medianBlur(gray,5)
        edges=cv2.adaptiveThreshold(greys,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color=cv2.bilateralFilter(frame,9,250,250)
    
        if first_frame is None:
            first_frame=gray
            continue
        cartoos=cv2.bitwise_and(color,color,mask=edges)
        delta_frame=cv2.absdiff(first_frame,gray) 
        can_edge=cv2.Canny(gray,100,200)
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        i=0
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("canny",can_edge)
        cv2.imshow("cartoos",cartoos)
        cv2.imshow("thresh Frame",thresh_frame)
        cv2.imshow("Threshold Frame",thresh)
        cv2.imshow("Color Frame",frame)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video.release()
    
    cv2.destroyAllWindows()





def face_blur():
    
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    


    
    video=cv2.VideoCapture(0)
    
    
    first_frame=None
    
    while True:
        check,frame=video.read()
        
        

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        if first_frame is None:
            first_frame=gray
            continue
        delta_frame=cv2.absdiff(first_frame,gray) 
        can_edge=cv2.Canny(gray,100,200)
        
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        i=0
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("HSV Frame",hsv_frame)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("canny",can_edge)
        #cv2.imshow("binary",BIT)
        cv2.imshow("thresh Frame",thresh_frame)
        cv2.imshow("Threshold Frame",thresh)
        cv2.imshow("Color Frame",frame)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video.release()
    
    cv2.destroyAllWindows()

def face_blurs():
    
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture("face.mp4")
    first_frame=None
    
    while True:
        check,frame=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        our_frame=frame[:,:,::-1].copy()
        img_gray=cv2.cvtColor(our_frame,cv2.COLOR_BGR2GRAY)
        img_invert=cv2.bitwise_not(img_gray)
        blur=cv2.GaussianBlur(img_invert,(21,21),0)
        invertedblur=cv2.bitwise_not(blur)
        greys=cv2.medianBlur(gray,5)
        edges=cv2.adaptiveThreshold(greys,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color=cv2.bilateralFilter(frame,9,250,250)
        if first_frame is None:
            first_frame=gray
            continue
        cartoos=cv2.bitwise_and(color,color,mask=edges)
        sketch=cv2.divide(img_gray,invertedblur,scale=256.0)
        delta_frame=cv2.absdiff(first_frame,gray) 
        can_edge=cv2.Canny(gray,100,200)
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        i=0
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("canny",can_edge)
        cv2.imshow("sketch",sketch)
        cv2.imshow("cartoos",cartoos)
        cv2.imshow("thresh Frame",thresh_frame)
        cv2.imshow("Threshold Frame",thresh)
        cv2.imshow("Color Frame",frame)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    

def move_object():
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture(0)
    first_frame=None
    
    while True:
        check,frame=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        our_frame=frame[:,:,::-1].copy()
        img_gray=cv2.cvtColor(our_frame,cv2.COLOR_BGR2GRAY)
        img_invert=cv2.bitwise_not(img_gray)
        blur=cv2.GaussianBlur(img_invert,(21,21),0)
        invertedblur=cv2.bitwise_not(blur)
        
        if first_frame is None:
            first_frame=gray
            continue
        sketch=cv2.divide(img_gray,invertedblur,scale=256.0)
        delta_frame=cv2.absdiff(first_frame,gray) 
        can_edge=cv2.Canny(gray,100,200)
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        i=0
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            #img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("canny",can_edge)
        cv2.imshow("sketch",sketch)
        cv2.imshow("thresh Frame",thresh_frame)
        cv2.imshow("Threshold Frame",thresh)
        cv2.imshow("Color Frame",frame)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
def move_objects():
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture("face.mp4")
    
    first_frame=None
    
    while True:
        check,frame=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        our_frame=frame[:,:,::-1].copy()
        img_gray=cv2.cvtColor(our_frame,cv2.COLOR_BGR2GRAY)
        img_invert=cv2.bitwise_not(img_gray)
        blur=cv2.GaussianBlur(img_invert,(21,21),0)
        invertedblur=cv2.bitwise_not(blur)
        greys=cv2.medianBlur(gray,5)
        edges=cv2.adaptiveThreshold(greys,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color=cv2.bilateralFilter(frame,9,250,250)
        if first_frame is None:
            first_frame=gray
            continue
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cartoos=cv2.bitwise_and(color,color,mask=edges)
        sketch=cv2.divide(img_gray,invertedblur,scale=256.0)
        delta_frame=cv2.absdiff(first_frame,gray) 
        can_edge=cv2.Canny(gray,100,200)
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        i=0
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            #img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("HSV frame",hsv)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("canny",can_edge)
        cv2.imshow("sketch",sketch)
        cv2.imshow("cartoos",cartoos)
        cv2.imshow("thresh Frame",thresh_frame)
        cv2.imshow("Threshold Frame",thresh)
        cv2.imshow("Color Frame",frame)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

st.markdown("""
<style>
.stButton>button{

        color:#a8e063;
        background-color:#000428;
    }


    


""",unsafe_allow_html=True)

custom_title="""
<div style="font-size:35px;font-weight:bolder;background-color:orange;padding:10px;
border-radius:10px;border:5px solid #3a1c71;text-align:center;">
<span style='color:blue'>F</span>
<span style='color:pink'>A</span>
<span style='color:green'>C</span>
<span style='color:dark blue'>E</span>
<span style='color:magenta'>-</span>
<span style='color:light green'>D</span>
<span style='color:black'>E</span>
<span style='color:red'>T</span>
<span style='color:magenta'>E</span>
<span style='color:green'>C</span>
<span style='color:black'>T</span>
<span style='color:blue'>I</span>
<span style='color:pink'>O</span>
<span style='color:green'>N</span>
<span style='color:purple'>-</span>
<span style='color:dark blue'>W</span>
<span style='color:light blue'>E</span>
<span style='color:black'>B</span>
<span style='color:red'>A</span>
<span style='color:pink'>P</span>
<span style='color:blue'>P</span>

</div>



"""

def skin_tracker():
    def nothing(x):
        pass
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    video.set(3,640)
    video.set(4,480)
    img=np.zeros((300,512,3),np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)
    switch='0:OFF \n1:ON'
    cv2.createTrackbar(switch,'image',0,1,nothing)
    #b1=int(input("enter number"))
    #g1=int(input("enter number"))
    #r1=int(input("enter number"))
    while(True):
        check,frame=video.read()
        check,frames=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        min_yCrCb=np.array([1 ,20 ,75],np.uint8)
        max_yCrCb=np.array([30, 255, 255],np.uint8)
        imageYCrCb=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        skinRegionyCrCb=cv2.inRange(imageYCrCb,min_yCrCb,max_yCrCb)
        skinyCrCb=cv2.add(frame,frame,mask=skinRegionyCrCb)
        r=cv2.getTrackbarPos('R','image')
        g=cv2.getTrackbarPos('G','image')
        b=cv2.getTrackbarPos('B','image')
        s=cv2.getTrackbarPos(switch,'image')
        frame[skinRegionyCrCb>0]=(b,g,r)
        i=0
        for x,y,w,h in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            #img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow("gray",gray)
        
        cv2.imshow("original",frames)
        cv2.imshow("red skin image",frame)
        #cv2.imshow("original image",images)
        cv2.imshow("skin",skinyCrCb)
        cv2.imshow('image',img)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
        if s==0:
            img[:]=0
        else:
            img[:]=[b,g,r]
            #frame[skinRegionyCrCb>0]=(b,g,r)
    cv2.destroyAllWindows()

def skin_trackers():
    def nothing(x):
        pass
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video=cv2.VideoCapture("face.mp4")
    video.set(3,640)
    video.set(4,480)
    img=np.zeros((300,512,3),np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)
    switch='0:OFF \n1:ON'
    cv2.createTrackbar(switch,'image',0,1,nothing)
    #b1=int(input("enter number"))
    #g1=int(input("enter number"))
    #r1=int(input("enter number"))
    while(True):
        check,frame=video.read()
        check,frames=video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        min_yCrCb=np.array([1 ,20 ,75],np.uint8)
        max_yCrCb=np.array([30, 255, 255],np.uint8)
        imageYCrCb=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        skinRegionyCrCb=cv2.inRange(imageYCrCb,min_yCrCb,max_yCrCb)
        skinyCrCb=cv2.add(frame,frame,mask=skinRegionyCrCb)
        r=cv2.getTrackbarPos('R','image')
        g=cv2.getTrackbarPos('G','image')
        b=cv2.getTrackbarPos('B','image')
        s=cv2.getTrackbarPos(switch,'image')
        frame[skinRegionyCrCb>0]=(b,g,r)
        i=0
        for x,y,w,h in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            #img[y:y+h,x:x+w]=cv2.medianBlur(img[y:y+h,x:x+w],35)
            i=i+1
            cv2.putText(frame,"face count"+str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow("gray",gray)
        
        cv2.imshow("original",frames)
        cv2.imshow("red skin image",frame)
        #cv2.imshow("original image",images)
        cv2.imshow("skin",skinyCrCb)
        cv2.imshow('image',img)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
        if s==0:
            img[:]=0
        else:
            img[:]=[b,g,r]
            #frame[skinRegionyCrCb>0]=(b,g,r)
    cv2.destroyAllWindows()

def get_image_download_link(img):
    buffered=BytesIO()
    img.save(buffered,format="JPEG")
    img_str=base64.b64encode(buffered.getvalue()).decode()
    href=f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
    return href






def main():
    
    components.html(custom_title)
    
    st.title( "Using the Haar cascades classifiers:sunflower:")
    components.html("""

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-BmbxuPwQa2lc/FVzBcNJ7UAyJxM6wuqIj61tLrc4wSX0szH/Ev+nYRRuWlolflfl" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/js/bootstrap.bundle.min.js" integrity="sha384-b5kHyXgcpbZJO/tY9Ul7kGkf1S0CWuKcCD38l8YkeH8z8QjE0GmW1gYU5S9FOnJ0" crossorigin="anonymous"></script>

    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.6.0/dist/umd/popper.min.js" integrity="sha384-KsvD1yqQ1/1+IA7gi3P0tyJcT3vR+NdBTt13hSJ2lnve8agRGXTTyNaBYmCR/Nwi" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/js/bootstrap.min.js" integrity="sha384-nsg8ua9HAw1y0W1btsyWgBklPnCUAFLuTMS2G72MMONqmOymq585AcH49TLBQObG" crossorigin="anonymous"></script>

<div id="carouselExampleControls" class="carousel slide" data-bs-ride="carousel">
  <div class="carousel-inner">
    <div class="carousel-item active">
      <img src="https://www.israel21c.org/wp-content/uploads/2020/04/shutterstock_731158624.jpg" class="d-block w-100">
    </div>
    <div class="carousel-item">
      <img src="https://www.anotherwindowsblog.com/wp-content/uploads/2010/07/139-Facial-Login-Featured.jpg" class="d-block w-100" alt="...">
    </div>
    <div class="carousel-item">
      <img src="https://findface.pro/wp-content/uploads/2019/03/4.jpg" class="d-block w-100" alt="...">
    </div>
    <div class="carousel-item">
      <img src="https://miro.medium.com/max/3200/1*uBEAhu5ngPMnEuX3rk5N3g.png" class="d-block w-100" alt="...">
    </div>
  </div>
  <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleControls"  data-bs-slide="prev">
    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleControls"  data-bs-slide="next">
    <span class="carousel-control-next-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>
    
    """,height=800,)
    
    
    

    activities=["Home","About","face_blur","face_detect","skin_tracker"]
    choice=st.sidebar.selectbox("pick something Fun",activities)
    
    if choice == "Home":
        st.balloons()
        

        #st.markdown("**Go To About Section:sunglasses:**")
        abou_temp="""
        <div style="background-color:yellow;padding:5px;border-radius:5px">
        <h1 style"color:white;text-align:center;">Go to the About Section for more.............</h1>
        </div>
        """
        components.html(abou_temp)
        
        
        #if st.button("About"):
                #about()
        
        #if st.button("face_blur"):
                #face_blur()

        #move_object()
        #if st.button("skin_tracker"):
            #skin_tracker()
        


        image_file=st.file_uploader("Upload image",type=['jpeg','png','jpg','webp','jfif'])
        
        if image_file is not None:
            image=Image.open(image_file)
           
            st.write("**ORIGINAL IMAGE**")
            st.image(image,width=800)
            if st.button("Process"):
                result_img,result_faces=detect(image=image)
                st.write("**FACE DETECTION**")
                st.image(result_img,width=800)
                result1=Image.fromarray(result_img)
                st.markdown(get_image_download_link(result1),unsafe_allow_html=True)
                st.success("Found {} faces".format(len(result_faces)))
                result_img1,result_faces1=mask(our_image=image)
                
                    
                st.write("**BGR IMAGE**")
                st.image(result_img1,width=800)
                result2=Image.fromarray(result_img1)
                st.markdown(get_image_download_link(result2),unsafe_allow_html=True)
                st.write("**SKETCH IMAGE**")
                st.image(result_faces1,width=800)
                result3=Image.fromarray(result_faces1)
                st.markdown(get_image_download_link(result3),unsafe_allow_html=True)
                st.write("**CARTOON IMAGE**")
                res_img1,result_fac3=cartoon(t_image=image)
                st.image(result_fac3,width=800)
                result4=Image.fromarray(result_fac3)
                st.markdown(get_image_download_link(result4),unsafe_allow_html=True)
            
                result_img2,result_faces2=cannys(my_image=image)
                st.write("**CANNY IMAGE**")
                st.image(result_faces2,width=800)
                result5=Image.fromarray(result_faces2)
                st.markdown(get_image_download_link(result5),unsafe_allow_html=True)
                result_img3,result_face_blur=blur(image=image)
                st.write("**Blur Image**")
                st.image(result_img3,width=800)
                result6=Image.fromarray(result_img3)
                st.markdown(get_image_download_link(result6),unsafe_allow_html=True)
                result_img4,result_hsv=hsv(image1=image)
                st.write("**HSV IMAGE**")
                st.image(result_hsv,width=800)
                result=Image.fromarray(result_hsv)
                st.markdown(get_image_download_link(result),unsafe_allow_html=True)
                st.success("Found {} faces".format(len(result_face_blur)))
            
                

        enhance_type=st.sidebar.radio("Enhance Type",["original","gray-scale","contrast","brightness","blurring"])
        if enhance_type=='gray-scale':
            new_img=np.array(image.convert('RGB'))
            img=cv2.cvtColor(new_img,1)
            g_rate=st.sidebar.slider("gray",0.5,7.5)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.image(gray,width=800)
            
        

        elif enhance_type=='brightness':
            c_rate=st.sidebar.slider("Brightness",0.5,3.5)
            enhancer=ImageEnhance.Brightness(image)
            img_output=enhancer.enhance(c_rate)
            st.image(img_output,width=800)
            

        elif enhance_type=='contrast':
            c_rate=st.sidebar.slider("contrast",0.5,3.5)
            enhancer=ImageEnhance.Contrast(image)
            img_outputs=enhancer.enhance(c_rate)
            st.image(img_outputs,width=800)
            

        elif enhance_type=='blurring':
            new_img=np.array(image.convert('RGB'))
            blur_rate=st.sidebar.slider("Brightness",0.5,7.5)
            img=cv2.cvtColor(new_img,1)
            blur_img=cv2.GaussianBlur(img,(11,11),blur_rate)
            st.image(blur_img,width=800)
            
        #elif enhance_type=='original':
            #st.write("**Original Image**")

            #st.image(image,use_column_width=True)
        #else:
            #st.image(image,use_column_width=True)


            
                
    
    elif choice == "About":
        
        st.balloons()
        about()

    elif choice=="face_blur":
        st.balloons()
        #img=Image.open("blurr.jpg")
        #st.image(img,width=1100,caption="About face Detection")
        if st.button("process through webcam"):
            face_blur()
        if st.button("process through video"):
            face_blurs()
        
            
        


    elif choice=="face_detect":
        st.balloons()
        #img=Image.open("det.png")
        #st.image(img,width=1100,caption="About face Detection")
        if st.button("process through webcam"):
            move_object()
        if st.button("process through video"):
            move_objects()
        

        if st.button("cartoon"):
            cartoon_vid()
    elif choice=="skin_tracker":
        st.balloons()
        #img=Image.open("fac1.jpeg")
        #st.image(img,width=900,caption="About face Detection")
        if st.button("process through webcam"):
            skin_tracker()
        #if st.button("process through video"):
            #skin_trackers()

if __name__ == "__main__":
    main()


