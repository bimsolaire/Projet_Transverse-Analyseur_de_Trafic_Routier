import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("periph.mp4")

# initialize count
count = 0
center_point_prev_frame = []

tracking_object = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Center point current frame
    center_point_cur_frame = []

    # Detect Objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w)/2)
        cy = int((y + y + h)/2)
        center_point_cur_frame.append((cx, cy))
        print("FRAME N°", count, "BOX", x, y, w, h)
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_point_cur_frame:
            for pt2 in center_point_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 50:
                    tracking_object[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_object.copy()
        center_point_cur_frame_copy = center_point_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_point_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 50:
                    tracking_object[object_id] = pt
                    object_exists = True
                    if pt in center_point_cur_frame:
                        center_point_cur_frame.remove(pt)
                    continue

            # Remove the IDs
            if not object_exists:
                tracking_object.pop(object_id)

        # Add new IDs found
        for pt in center_point_cur_frame:
            tracking_object[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_object.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking Objects")
    print(tracking_object)

    print("CUR FRAME")
    print(center_point_cur_frame)

    print("PREV FRAME")
    print(center_point_prev_frame)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_point_prev_frame = center_point_cur_frame.copy()

    key = cv2.waitKey(1)

    if key == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()