from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

# Change this after calibration
REFERENCE_DISTANCE = 2.0
REFERENCE_HEIGHT = 300.0

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        # We are only testing people for now
        if object_name != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Height of bounding box in pixels
        object_height = y2 - y1

        if object_height > 0:

            estimated_distance = (
                REFERENCE_DISTANCE * REFERENCE_HEIGHT
            ) / object_height

            print(
                f"Person height: {object_height:.0f}px | "
                f"Estimated distance: {estimated_distance:.2f} m"
            )

    cv2.imshow("Distance Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()