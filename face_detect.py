import cv2 as cv

img = cv.imread('pics/elon.jpg')
img = cv.resize(img,(img.shape[1]//2,img.shape[0]//2))
cv.imshow("Elon",img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'No. of Faces found are : {len(face_rect)}')

for (x,y,w,h) in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces',img)

cv.waitKey(0)