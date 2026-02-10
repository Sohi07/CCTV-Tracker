import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json

# Setup 
model = YOLO("yolov5nu.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("video/Recording_2.mp4")
assert cap.isOpened(), "Cannot open video"

# Frame Settings
frame_width = 480
frame_height = 270
FRAME_AREA = frame_width * frame_height

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 0 and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    active_ids = set()

    # -------- Draw Tracked Points --------
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        active_ids.add(track_id)

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Centroid (head point approx)
        cx = (x1 + x2) // 2
        cy = y1 + int(0.15 * (y2 - y1))

        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    #People Counting 
    people_count = len(active_ids)

    # Density Calculation
    density = people_count / FRAME_AREA

    # Risk Classification 
    if density < 0.0005:
        crowd_status = "SAFE"
        status_color = (0, 255, 0)
    elif density < 0.001:
        crowd_status = "MODERATE"
        status_color = (0, 255, 255)
    else:
        crowd_status = "DANGEROUS"
        status_color = (0, 0, 255)

    # -------- Alert Message --------
    if crowd_status == "DANGEROUS":
        cv2.putText(frame, "⚠ HIGH CROWD DENSITY ALERT",
                    (100, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Save Frame for Dashboard 
    cv2.imwrite("latest_frame.jpg", frame)

    #  Export Data for Streamlit 
    data = {
        "count": people_count,
        "density": density,
        "status": crowd_status
    }

    with open("live_data.json", "w") as f:
        json.dump(data, f)

    display_frame = cv2.resize(frame, (700, 450))
    cv2.imshow("Crowd Safety Monitoring System", display_frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
