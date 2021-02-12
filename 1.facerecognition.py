# #save image at this path
# import os
# def assure_path_exists(path):
#     dir = os.path.dirname(path)
#     if not os.path.exists(dir):
#         os.makedirs(dir)
# assure_path_exists("dataset/")


#import opencv library  for image processing
import cv2

#detect objet in video streming using haarcascade 
frontal_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#start camera
cam = cv2.VideoCapture(0)
#set display of image processing
cam.set(4, 600)
cam.set(4,600)

# For each person, one face id
face_id = 1

# Initialize sample face image
count = 0



while(True):
    ret , photo = cam.read()
    print(ret)
    #change color of image into gray
    grey = cv2.cvtColor(photo , cv2.COLOR_BGR2GRAY)
    
    #start framing on image 
    faces =frontal_cascade.detectMultiScale(photo, 1.3,5)
    
    #loop for each faces
    for(x,y,w,h) in faces:
        cv2.rectangle(photo, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", photo[y:y+h,x:x+w])
    
    cv2.imshow('frame', photo)
    if cv2.waitKey(100)==27:
            break
    # If image taken reach 200, stop taking video
    elif count>50:
        break
#stop camera
cam.release()
cv2.destroyAllWindows()
        
    
