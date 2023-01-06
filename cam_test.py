import cv2 as cv

stream = cv.VideoCapture(0) # local camera

while True:
    isTrue, frame = stream.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'): # press 'd'to stop
        break

stream.release()
cv.destroyAllWindows()

cv.waitKey(0)