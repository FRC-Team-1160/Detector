import detect
import cv2 as cv
import time

stream = cv.VideoCapture(1)

while 1:
    # On "q" keypress twice, exit
    if(cv.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    ret, img = stream.read()
    image = detect.infer(img)
    cv.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

stream.release()
cv.destroyAllWindows()

cv.waitKey(0)