import cv2
import mediapipe as mp
import random
import numpy as np

mp_face_detection = mp.solutions.face_detection

def get_centroid(bbox):
    # bbox: [xmin, ymin, width, height] normalized
    xmin, ymin, w, h = bbox
    return (xmin + w / 2, ymin + h / 2)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def crop_face_from_frame(frame, bbox, pad=0.1):
    h, w, _ = frame.shape
    xmin, ymin, bw, bh = bbox

    # Apply padding
    xmin_p = max(0, xmin - bw * pad)
    ymin_p = max(0, ymin - bh * pad)
    xmax_p = min(1, xmin + bw * (1 + pad))
    ymax_p = min(1, ymin + bh * (1 + pad))

    x1 = int(xmin_p * w)
    y1 = int(ymin_p * h)
    x2 = int(xmax_p * w)
    y2 = int(ymax_p * h)

    return frame[y1:y2, x1:x2]

def smooth_bbox(prev_bbox, new_bbox, alpha=0.6):
    return [
        alpha * p + (1 - alpha) * n
        for p, n in zip(prev_bbox, new_bbox)
    ]

def significant_change(prev_bbox, new_bbox, threshold=0.01):
    dw = abs(prev_bbox[2] - new_bbox[2])
    dh = abs(prev_bbox[3] - new_bbox[3])
    return dw > threshold or dh > threshold

def track_face():
    cap = cv2.VideoCapture(0)
    tracked_centroid = None
    prev_bbox = None

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                centroids = []
                bboxes = []

                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    bbox = [bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height]
                    cent = get_centroid(bbox)
                    centroids.append(cent)
                    bboxes.append(bbox)

                if tracked_centroid is None:
                    # First time: choose a random face
                    idx = random.randint(0, len(centroids) - 1)
                else:
                    # Find the closest face to previous tracked centroid
                    distances = [euclidean_distance(c, tracked_centroid) for c in centroids]
                    idx = int(np.argmin(distances))

                current_bbox = bboxes[idx]
                tracked_centroid = centroids[idx]

                if prev_bbox is None:
                    smoothed_bbox = current_bbox
                elif significant_change(prev_bbox, current_bbox):
                    smoothed_bbox = smooth_bbox(prev_bbox, current_bbox, alpha=0.6)
                else:
                    smoothed_bbox = prev_bbox  # hold previous

                prev_bbox = smoothed_bbox

                # Draw tracking box
                h, w, _ = frame.shape
                x1 = int(smoothed_bbox[0] * w)
                y1 = int(smoothed_bbox[1] * h)
                x2 = int((smoothed_bbox[0] + smoothed_bbox[2]) * w)
                y2 = int((smoothed_bbox[1] + smoothed_bbox[3]) * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop and show face
                cropped = crop_face_from_frame(frame, smoothed_bbox, pad=0.1)
                if cropped.size > 0:
                    cv2.imshow("Tracked Face", cropped)

                # Draw centroid
                cx_px = int(tracked_centroid[0] * w)
                cy_px = int(tracked_centroid[1] * h)
                cv2.circle(frame, (cx_px, cy_px), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Centroid: ({cx_px}, {cy_px})", (cx_px + 10, cy_px),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            else:
                tracked_centroid = None
                prev_bbox = None  # Reset if no face

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_face()
