import cv2
import os

cascadeFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascadeEye = cv2.CascadeClassifier('haarcascade_eye.xml')
SaveVideoPatch = 'video/video_1/'
prefix = '2_'
Play = True
VideoPatch = 'video_sources/SeroVolk/1.mp4'
vcap = cv2.VideoCapture(VideoPatch)
counter = 802

while (vcap.isOpened()) and Play:
    retCap, frameCap = vcap.read()
    grayColor = cv2.cvtColor(frameCap, cv2.COLOR_BGR2GRAY)
    foundFaces = cascadeFace.detectMultiScale(grayColor, scaleFactor=1.2, minNeighbors=3, minSize=(100, 100))
    
    for (x, y, w, h) in foundFaces:
        grayRoi = grayColor[y:y + h, x:x + w]
        foundEyes = cascadeEye.detectMultiScale(grayRoi, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10))
        if len(foundEyes) > 0:
            name = prefix + str(counter) + '.jpg'
            counter += 1
            size = (100, 100)
            output = cv2.resize(grayRoi, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(SaveVideoPatch, name), output)
            cv2.rectangle(frameCap, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("camera", frameCap)
    if cv2.waitKey(10) == 27:
        Play = False
        
vcap.release()
cv2.destroyAllWindows()
