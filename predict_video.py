import os
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("test1.mp4")
# cap = cv2.VideoCapture(0)


model_path = os.path.join(
    'runs', 'detect', 'train5', 'weights', 'best.pt'
)
model = YOLO(model_path)

threshold = 0.5
offset = 5  

count = {
    "bottle": 0,
    "cap": 0
}

crossed_ids = {
    "bottle": set(),
    "cap": set()
}

line_x = 700

output_path = "output_counted1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if not results or results[0].boxes is None:
        cv2.imshow('out', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
        continue

    result = results[0]

    for box in result.boxes:
        if box.id is None:
            continue

        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = box.conf.item()
        class_name = result.names[int(box.cls[0].item())]

        if score < threshold:
            continue

        if class_name == "bottle":
            color = (255, 0, 0)
        elif class_name == "cap":
            color = (255, 200, 0)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        text = class_name.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, (x2 - x1) / 150)
        thickness = 1
        text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]

        if class_name == "bottle":
            tx, ty = x1, y1 - 5
        else:
            tx, ty = x1, y2 + text_h + 5

        rect_start = (tx - 2, ty - text_h - 4)
        rect_end = (tx + text_w + 2, ty + 2)
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 0), -1)  
        cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        front_x = x2
        if (line_x - offset) <= front_x <= (line_x + offset) and track_id not in crossed_ids[class_name]:
            count[class_name] += 1
            crossed_ids[class_name].add(track_id)

    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 0), 2)

    
    cv2.rectangle(frame, (0, 0), (350, 90), (0, 0, 0), -1)

    cv2.putText(frame, f"BOTTLE COUNT: {count['bottle']}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
    cv2.putText(frame, f"CAP COUNT: {count['cap']}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 1)

    cv2.imshow('out', frame)
    out.write(frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to", output_path)
