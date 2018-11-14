import numpy as np
import cv2
import pickle



face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() ##OpenCV Recognizer 
recognizer.read("trainer.yml")

labels={}
with open("labels.pkl",'rb')as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}



cap = cv2.VideoCapture(0)
pic_marker = 0
while(True):
    ret, frame = cap.read()
    ret, sf = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    #front_face
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #[Y+heigh, X+width]
        roi_color = frame[y:y+h, x:x+w]
        
        #recognize ? deep learning, keras, tensorflow, scikit learn
        id_, conf = recognizer.predict(roi_gray)

        if conf>=45: #and conf <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font,1,color,stroke, cv2.LINE_AA)


        
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
        cv2.imwrite(img_s+str(pic_marker)+".png",sf)
        pic_marker = pic_marker+1
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
