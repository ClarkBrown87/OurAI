import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 300)

faces = cv.CascadeClassifier('check_face.xml')

while True:
    success, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result = faces.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=4)

    for (x, y, w, h) in result:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)

    cv.imshow("Result", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break