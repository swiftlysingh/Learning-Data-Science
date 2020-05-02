import cv2
cap = cv2.VideoCapture(0)

while True:
    ret,image = cap.read()
    if ret:
        cv2.imshow("My First Camera",image)
        x,y,w,h = (100,120,150,150)
        cut = image[y:y+h,x:x+w]
        cv2.imshow("Chopped", cut)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        cv2.imwrite("classroon.jpg",image)

cap.release()
cv2.destroyAllWindows()



