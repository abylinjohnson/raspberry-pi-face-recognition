import cv2 as cv

capture = cv.VideoCapture(0)

for i in range(30):
    isTrue, img = capture.read()
    cv.imwrite(f'Faces/{i}.jpg',img)

