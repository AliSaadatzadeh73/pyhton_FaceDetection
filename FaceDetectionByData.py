import cv2
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier("../Resource/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../Resource/haarcascade_eye.xml")
smil_cascade= cv2.CascadeClassifier("../Resource/haarcascade_smil.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(1)

while (True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in face:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord1-start, ycord2-end)
        roi_color = frame[y:y+h, x:x+w]

       #recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            # print(id_)
            # print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            strok = 2
            cv2.putText(frame, name, (x,x), font, 1, color, strok,cv2.LINE_AA)

        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #BGR 0-255
        strok = 2
        end_cor_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y),(end_cor_x,end_cord_y),color,strok)
       #Display smil
        subitem = smil_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in subitem:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,16),2)
        #Display eyes
        eye = eye_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in eye:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)


    #Display the resulting fram
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything donr, relase the capture


cap.release()
cv2.destroyAllWindows()