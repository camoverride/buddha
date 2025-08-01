import cv2
import mediapipe as mp
import random
import numpy as np
from typing import Optional



mp_face_detection = mp.solutions.face_detection


def get_centroid(bbox : list) -> tuple[float, float]:
    """
    Computes the centroid of a bounding box around a face.

    Parameters
    ----------
    bbox : list[float]
        A list of normalized coordinates [xmin, ymin, width, height], 
        where values are in the range (0, 1).

    Returns
    -------
    tuple[float, float]
        The (x, y) centroid of the bounding box, in normalized coordinates.
    """
    xmin, ymin, w, h = bbox

    return (xmin + w / 2, ymin + h / 2)


def euclidean_distance(p1 : tuple[float, float],
                       p2 : tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two 2D points.

    Parameters
    ----------
    p1 : tuple[float, float]
        The (x, y) coordinates of the first point.
    p2 : tuple[float, float]
        The (x, y) coordinates of the second point.

    Returns
    -------
    float
        The Euclidean distance between p1 and p2.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def crop_face_from_frame(frame: np.ndarray,
                         bbox: list[float],
                         pad: float = 0.1) -> np.ndarray:
    """
    Crops a face region from the frame using a normalized
    bounding box with optional padding.

    Parameters
    ----------
    frame : np.ndarray
        The input image/frame as a NumPy array (H x W x C).
    bbox : list[float]
        Normalized bounding box [xmin, ymin, width, height],
        where all values are between 0 and 1.
    pad : float, optional
        Amount of padding to apply to all sides of the bounding box
        as a fraction of its size. Default is 0.1 (i.e. 10% padding).

    Returns
    -------
    np.ndarray
        The cropped face image as a NumPy array. If the padded area
        goes out of bounds, it will be clamped to the image dimensions.
    """
    h, w, _ = frame.shape
    xmin, ymin, bw, bh = bbox

    # Apply padding and clamp to [0, 1]
    xmin_p = max(0, xmin - bw * pad)
    ymin_p = max(0, ymin - bh * pad)
    xmax_p = min(1, xmin + bw * (1 + pad))
    ymax_p = min(1, ymin + bh * (1 + pad))

    # Convert normalized coordinates to pixel values
    x1 = int(xmin_p * w)
    y1 = int(ymin_p * h)
    x2 = int(xmax_p * w)
    y2 = int(ymax_p * h)

    return frame[y1:y2, x1:x2]


def smooth_bbox(prev_bbox: list[float],
                new_bbox: list[float],
                alpha: float = 0.6) -> list[float]:
    """
    Smooths a bounding box using exponential moving average (EMA)
    to reduce jitter between frames.

    Parameters
    ----------
    prev_bbox : list[float]
        The previous bounding box [xmin, ymin, width, height], normalized.
    new_bbox : list[float]
        The current bounding box [xmin, ymin, width, height], normalized.
    alpha : float, optional
        Smoothing factor. Higher values favor the previous bbox more.
        Range: 0.0 (no smoothing) to 1.0 (fully static). Default is 0.6.

    Returns
    -------
    list[float]
        A smoothed bounding box [xmin, ymin, width, height], normalized.
    """
    return [alpha * p + (1 - alpha) * n \
            for p, n in zip(prev_bbox, new_bbox)]


def significant_change(prev_bbox: list[float],
                       new_bbox: list[float],
                       threshold: float = 0.01) -> bool:
    """
    Determines whether the size of a bounding box has changed
    significantly between frames.

    Parameters
    ----------
    prev_bbox : list[float]
        The previous bounding box [xmin, ymin, width, height], normalized.
    new_bbox : list[float]
        The current bounding box [xmin, ymin, width, height], normalized.
    threshold : float, optional
        Minimum fractional change in width or height to be considered
        significant. Default is 0.01 (i.e. 1%).

    Returns
    -------
    bool
        True if the width or height changed more than the threshold,
        False otherwise.
    """
    dw = abs(prev_bbox[2] - new_bbox[2])
    dh = abs(prev_bbox[3] - new_bbox[3])

    return dw > threshold or dh > threshold


def track_face() -> None:
    """
    Starts webcam video capture and tracks a single face in real time.

    The system detects all faces in each frame using MediaPipe Face Detection,
    selects one face (either randomly at the beginning or by proximity in subsequent frames),
    and continuously crops and displays the face region with smoothing to avoid jitter.

    Features:
    - Tracks a consistent face across frames by comparing centroid distances.
    - Applies exponential moving average smoothing to the bounding box.
    - Adds optional padding around the cropped face.
    - Automatically resets if no faces are detected.

    Press 'q' to exit the webcam display.
    """
    cap = cv2.VideoCapture(0)
    tracked_centroid: Optional[tuple[float, float]] = None
    prev_bbox: Optional[tuple[float]] = None

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

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
                centroid = get_centroid(bbox)
                bboxes.append(bbox)
                centroids.append(centroid)

            if tracked_centroid is None:
                # First time: choose a face randomly
                idx = random.randint(0, len(centroids) - 1)
            else:
                # Track face closest to the previously tracked centroid
                distances = [euclidean_distance(c, tracked_centroid) for c in centroids]
                idx = int(np.argmin(distances))

            current_bbox = bboxes[idx]
            tracked_centroid = centroids[idx]

            if prev_bbox is None:
                smoothed_bbox = current_bbox
            elif significant_change(prev_bbox, current_bbox):
                smoothed_bbox = smooth_bbox(prev_bbox, current_bbox, alpha=0.6)
            else:
                smoothed_bbox = prev_bbox  # Hold previous box if no significant change

            prev_bbox = smoothed_bbox

            # Draw tracking box
            h, w, _ = frame.shape
            x1 = int(smoothed_bbox[0] * w)
            y1 = int(smoothed_bbox[1] * h)
            x2 = int((smoothed_bbox[0] + smoothed_bbox[2]) * w)
            y2 = int((smoothed_bbox[1] + smoothed_bbox[3]) * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop and display face
            cropped = crop_face_from_frame(frame, smoothed_bbox, pad=0.1)
            if cropped.size > 0:
                cv2.imshow("Tracked Face", cropped)

            # Draw centroid marker
            cx_px = int(tracked_centroid[0] * w)
            cy_px = int(tracked_centroid[1] * h)
            cv2.circle(frame, (cx_px, cy_px), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Centroid: ({cx_px}, {cy_px})", (cx_px + 10, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Reset tracking if no face is detected
            tracked_centroid = None
            prev_bbox = None

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()



if __name__ == "__main__":
    track_face()
