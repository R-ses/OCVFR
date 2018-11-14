import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
pic_marker = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    #front_face
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #[Y+heigh, X+width]
        roi_color = frame[y:y+h, x:x+w]
        
        #recognize ? deep learning, keras, tensorflow, scikit learn
        
        
        img_s = "" #Nombre ""+pic_marker+".png"

        #img_item = "my_image.png"
        #cv2.imwrite(img_item,roi_gray)

        #print_rectangle
        
        color = (255,0,0) #BGR 0 - 255
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height), color, stroke)

    if cv2.waitKey(20) & 0xFF == ord('p'):
        cv2.imwrite(img_s+str(pic_marker)+".png",roi_gray)
        pic_marker = pic_marker+1
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
