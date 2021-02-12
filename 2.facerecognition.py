#traning dataset


import cv2 , os
import numpy as np
from PIL import Image


fasecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImageAndLabel(path):
    ids = []
    facesample = []
    imagepaths  = [os.path.join(path ,f) for f in os.listdir(path)]
    for imagepath in imagepaths:
        
        Pil_image = Image.open(imagepath).convert("L")
        Pil_np = np.array(Pil_image,"uint8")
        
        id = int(os.path.split(imagepath)[-1][5])
        print(id)
        faces = fasecascade.detectMultiScale(Pil_np)
        
        for(x,y,w,h) in faces:
            
            facesample.append(Pil_np[y:y+h , x:x+w])
            ids.append(id)
    return facesample, ids


faces , ids =getImageAndLabel("dataset")
print(faces)
print(ids)

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.save('trainer/trainer.yml')

