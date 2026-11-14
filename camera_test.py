import cv2

print("Starting camera...")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

while True:
    success, frame = cap.read()

    print("Frame received:", success)

    if not success:
        print("ERROR: Could not read frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()