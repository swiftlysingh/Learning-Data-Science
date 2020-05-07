import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = np.load("faces.npy")
X,y_train = data[:,1:],data[:,0]

model = KNeighborsClassifier()
model.fit(X,y_train)


while True:
    ret,image = cap.read()
    if ret:
        faces = classifier.detectMultiScale(image)

        if len(faces)   > 0:

            sorted_faces = sorted(faces,key=lambda item:item[2]*item[3])

            x,y,w,h = sorted_faces[-1]

            cv2.rectangle(image, (x,y) , (x+w , y+h) , (0,0,0),2)

            cut = image[y:y+h,x:x+w]

            resized = cv2.resize(cut,(100,100))
            y_test_item = resized.mean(axis=2).flatten()

            output = model.predict([y_test_item])[0]

            cv2.putText(image,str(output),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(image, str(model.score(X,y_train)*100)+"%", (x+w, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    key = cv2.waitKey(1)
    cv2.imshow("My First Camera", image)
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

