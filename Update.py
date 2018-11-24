import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "train")

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)##OpenCV Recognizer



current_id = 0
label_ids = {}
y_labels = []
x_train= []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("JPEG"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","_").lower()
            print(label,path)
                                    
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            #print(label_ids)	
            #x_train.append(path)
            #y_labels.append(label)

            pil_image = Image.open(path).convert("L") #grayscale
            size = (550,550)
            final_image =pil_image.resize(size, Image.ANTIALIAS) #Ajuste de fotos
            image_array = np.array(final_image, "uint8") #datos de las imagenes para entrenar
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            #print(faces)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)



#print(y_labels)
#print(x_train)

with open("labels.pkl",'wb')as f:
    pickle.dump(label_ids, f)

recognizer.update(x_train, np.array(y_labels))
#recognizer.save("trainer.yml")




                
