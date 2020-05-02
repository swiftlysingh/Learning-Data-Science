import cv2
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret,image = cap.read()
    if ret:
        faces = classifier.detectMultiScale(image)
        cv2.imshow("My First Camera", image)

        if len(faces) >= 2:
            sorted_faces = sorted(faces,key=lambda item:item[2]*item[3])

            face1,face2 = sorted_faces[-2:]

            x1,y1,w1,h1 = face1
            x2, y2, w2, h2 = face2

            cut1 = image[y1:y1+h1,x1:x1+w1]
            cut2 = image[y2:y2+h2,x2:x2+w2]

            t_cut1 = cv2.resize(cut2,(cut1.shape[1],cut1.shape[0]))
            t_cut2 = cv2.resize(cut1,(cut2.shape[1],cut2.shape[0]))

            image[y2:y2 + h2, x2:x2 + w2] = t_cut2
            image[y1:y1 + h1, x1:x1 + w1] = t_cut1

            cv2.imshow("Swapped", image)
        else:
            cv2.imshow("Swapped",image)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        cv2.imwrite("classroon.jpg",image)

cap.release()
cv2.destroyAllWindows()



