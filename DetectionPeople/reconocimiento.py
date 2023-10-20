import cv2

cascade_classifier = cv2.CascadeClassifier('C:/Users/Alejandro Rojas/Documents/GitHub/cameraPython/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error a la conexion")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudieron capturar cuadros.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detección de Personas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
