import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import os

# --- Setup ---
model = YOLO('yolov5nu.pt')  # Light YOLOv5 Nano model
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("video/Recording.mp4")
assert cap.isOpened(), "Cannot open video"

# Output folder for saved suspect crops
os.makedirs("output/suspects", exist_ok=True)

# Mutable suspect ID
suspect_id = [0]

# Last save time for suspect snapshots
last_save_time = 0

# --- Background Input Thread ---
def input_thread():
    while True:
        try:
            new_id = int(input("Enter new suspect ID: "))
            suspect_id[0] = new_id
            print(f"Suspect ID updated to {new_id}")
        except ValueError:
            print("Invalid ID. Please enter a number.")

threading.Thread(target=input_thread, daemon=True).start()

# --- Main Tracking Loop ---
frame_count = 0
save_interval = 1.0  # seconds
frame_width = 640
frame_height = 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))  # Smaller frame

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 0 and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = int(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = (0, 0, 255) if track_id == suspect_id[0] else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize display frame
    display_frame = cv2.resize(frame, (700, 450))  # Smaller UI window
    cv2.imshow("Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # q or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
