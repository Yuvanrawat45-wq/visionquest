from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

speaking = False

def speak(text):
    global speaking

    speaking = True

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

    speaking = False


last_spoken = {}
cooldown = 3

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    # Get camera dimensions
    height, width, _ = frame.shape

    # Center of the camera
    camera_center = width / 2

    current_time = time.time()

    for box in results[0].boxes:

        # Object information
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Find center of the detected object
        object_center = (x1 + x2) / 2

        # Determine object position
        if object_center < width / 3:
            position = "on your left"

        elif object_center > (2 * width / 3):
            position = "on your right"

        else:
            position = "in front of you"

        # Create unique key for object + position
        detection_key = f"{object_name}_{position}"

        last_time = last_spoken.get(detection_key, 0)

        # Speak
        if current_time - last_time >= cooldown and not speaking:

            message = f"{object_name} {position}"

            print("Speaking:", message)

            last_spoken[detection_key] = current_time

            threading.Thread(
                target=speak,
                args=(message,),
                daemon=True
            ).start()

            break

    cv2.imshow("Vision Project - YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()