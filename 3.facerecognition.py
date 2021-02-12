#testing dataset

import cv2 , os

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')


facialharr = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
camera.set(4,600)
camera.set(4,600)

while(True):
    ret , photo = camera.read()
    gray = cv2.cvtColor(photo , cv2.COLOR_BGR2GRAY)
    faces = facialharr.detectMultiScale(photo,1.3,5)
    
    for(x,y,w,h) in faces:
        
        cv2.rectangle(photo, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

       
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id)
         
        if(Id[0] == 1 ):
            print('vivek')
        else:
            print('Unknown')

        # Put text describe who is in the picture
        cv2.rectangle(photo, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        if(Id[0]==1):
            cv2.putText(photo, 'vivek', (x,y-40), font, 2, (255,255,255), 3)

        
    cv2.imshow("image" ,photo)
    if cv2.waitKey(1)==27:
        break


camera.release()
cv2.destroyAllWindows()
