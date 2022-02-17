import cv2
from keras.models import load_model
model = load_model("mymodel2.h5")
import numpy as np
frameWidth= 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture("http://192.168.1.15:8080/video")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

labels = open("labels.csv",'r').readlines()
labels = labels[1::]
lbl=[]
for label in labels:
    lbl.append(label.split(',')[1].rstrip('\n'))

while True:

# READ IMAGE
    success, imgOrignal = cap.read()
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = np.sum(img/3, axis=2, keepdims = True)

    # cv2.imshow("Processed Image", img)

    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img.reshape(1, 32, 32, 1))
    

    y_classes = [np.argmax(element) for element in predictions]
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
    # print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(y_classes) + " " + str(lbl[y_classes[0]]), (120, 35), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
