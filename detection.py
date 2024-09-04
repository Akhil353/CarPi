import cv2
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

dist_obj = distance_calculation.DistanceCalculation(names=model.model.names, view_img=True)

clicked = False

def mouse_callback(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True

cv2.namedWindow('Distance Calculation')
cv2.setMouseCallback('Distance Calculation', mouse_callback)

def calculate_pairwise_distances(tracks):
    distances = []
    for i, track_a in enumerate(tracks):
        for j, track_b in enumerate(tracks):
            if i >= j:
                continue
            center_a = ((track_a['box'][0] + track_a['box'][2]) / 2, (track_a['box'][1] + track_a['box'][3]) / 2)
            center_b = ((track_b['box'][0] + track_b['box'][2]) / 2, (track_b['box'][1] + track_b['box'][3]) / 2)
            distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
            distances.append((center_a, center_b, distance))
    return distances

while True:
    success, frame = cap.read()
    if not success:
        break

    tracks = model.track(frame, persist=True)


    frame = dist_obj.start_process(frame, tracks)

    cv2.imshow('Distance Calculation', frame)

    if clicked:
        results = model(frame)
        tracks = results[0].boxes.data.cpu().numpy()

        pairwise_distances = calculate_pairwise_distances(tracks)

        for (center_a, center_b, distance) in pairwise_distances:
            cv2.line(frame, (int(center_a[0]), int(center_a[1])), (int(center_b[0]), int(center_b[1])), (0, 255, 0), 2)
            mid_point = ((center_a[0] + center_b[0]) // 2, (center_a[1] + center_b[1]) // 2)
            cv2.putText(frame, f"{distance:.2f} px", (int(mid_point[0]), int(mid_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        clicked = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
