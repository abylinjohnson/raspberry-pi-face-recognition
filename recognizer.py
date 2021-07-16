import numpy as np 
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Elon','Tom Cruise']
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
capture = cv.VideoCapture(0)
while True:
    isTrue, img = capture.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray, 1.1, 3)

    for (x,y,w,h) in face_rect:
        faces_roi = gray[y:y+h,x:x+h]
        labels, confidence = face_recognizer.predict(faces_roi)
        if confidence > 50:
            print(f'Label =  {people[labels]} with a confidence {confidence}')
            cv.putText(img, str(people[labels])+str(confidence),(x,y-5), cv.FONT_HERSHEY_COMPLEX,1.0, (0,0,255),thickness=1 )
            cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=1)
    cv.imshow("Detected Face", img)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
        
capture.release()
cv.destroyAllWindows()      