from collections import deque
import cv2
from mediapipe.python.solutions import face_detection as mp_face_detection
import numpy as np
import os
import platform
import random
import re
import subprocess
import threading
from typing import Optional
import uuid
import yaml



def get_centroid(
    bbox : tuple[float, float, float, float]
    ) -> tuple[float, float]:
    """
    Computes the centroid of a normalized bounding box with
    coordinates relative to the image dimensions, where (0,0)
    is the top-left and (1,1) is the bottom-right.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        A tuple of normalized coordinates [xmin, ymin, width, height], 
        where values are in the range (0, 1).

    Returns
    -------
    tuple[float, float]
        The (x, y) centroid of the bounding box, in normalized coordinates.
    """
    xmin, ymin, w, h = bbox

    return (xmin + w / 2, ymin + h / 2)


def euclidean_distance(
    p1 : tuple[float, float],
    p2 : tuple[float, float]
    ) -> float:
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


def crop_face_from_frame(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad: float = 0.1
    ) -> np.ndarray:
    """
    Crops a face region from the frame using a normalized
    bounding box with optional padding.

    Parameters
    ----------
    frame : np.ndarray
        The input image/frame as a NumPy array (H x W x C).
    bbox : tuple[float, float, float, float]
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
    # NOTE: the padding expands the box by a fraction of its size on all sides.
    xmin_p = max(0, xmin - bw * pad)
    ymin_p = max(0, ymin - bh * pad)
    xmax_p = min(1, xmin + bw * (1 + pad))
    ymax_p = min(1, ymin + bh * (1 + pad))

    # Convert normalized coordinates to pixel values.
    x1 = int(xmin_p * w)
    y1 = int(ymin_p * h)
    x2 = int(xmax_p * w)
    y2 = int(ymax_p * h)

    return frame[y1:y2, x1:x2]


def smooth_bbox(
    prev_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    alpha: float = 0.6
    ) -> tuple[float, float, float, float]:
    """
    Smooths a bounding box using exponential moving average (EMA)
    to reduce jitter between frames.

    Parameters
    ----------
    prev_bbox : tuple[float, float, float, float]
        The previous bounding box [xmin, ymin, width, height], normalized.
    new_bbox : tuple[float, float, float, float]
        The current bounding box [xmin, ymin, width, height], normalized.
    alpha : float, optional
        Smoothing factor. Higher values favor the previous bbox more.
        Range: 0.0 (no smoothing) to 1.0 (fully static). Default is 0.6.

    Returns
    -------
    tuple[float, float, float, float]
        A smoothed bounding box [xmin, ymin, width, height], normalized.
    """
    return (alpha * prev_bbox[0] + (1 - alpha) * new_bbox[0],
            alpha * prev_bbox[1] + (1 - alpha) * new_bbox[1],
            alpha * prev_bbox[2] + (1 - alpha) * new_bbox[2],
            alpha * prev_bbox[3] + (1 - alpha) * new_bbox[3])


def smooth_point(prev_point: tuple[float, float],
                 new_point: tuple[float, float],
                 alpha: float = 0.6) -> tuple[float, float]:
    """
    Smooths a 2D point using exponential moving average (EMA).

    Parameters
    ----------
    prev_point : tuple[float, float]
        The previous point (x, y), normalized.
    new_point : tuple[float, float]
        The new point (x, y), normalized.
    alpha : float, optional
        Smoothing factor. Higher values favor the previous point.

    Returns
    -------
    tuple[float, float]
        The smoothed point.
    """
    smoothed_x = alpha * prev_point[0] + (1 - alpha) * new_point[0]
    smoothed_y = alpha * prev_point[1] + (1 - alpha) * new_point[1]
    return (smoothed_x, smoothed_y)


def significant_change(
    prev_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    threshold: float = 0.01
    ) -> bool:
    """
    Determines whether the size of a bounding box has changed
    significantly between frames.

    Parameters
    ----------
    prev_bbox : tuple[float, float, float, float]
        The previous bounding box [xmin, ymin, width, height], normalized.
    new_bbox : tuple[float, float, float, float]
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


def crop_with_aspect_ratio(
    frame: np.ndarray,
    centroid: tuple[float, float],
    bbox: tuple[float, float, float, float],
    target_aspect_ratio: float,
    relative_height: float
    ) -> np.ndarray:
    """
    Crops an image around a face using a fixed aspect ratio while scaling
    with the detected face size.

    The crop is centered on the face's centroid and sized based on the
    height of the bounding box. The width is then calculated to maintain
    the specified aspect ratio.

    Parameters
    ----------
    frame : np.ndarray
        The input image/frame as a NumPy array (H x W x C).
    centroid : tuple[float, float]
        The (x, y) centroid of the face in normalized coordinates [0, 1].
    bbox : tuple[float, float, float, float]
        Normalized bounding box [xmin, ymin, width, height].
    target_aspect_ratio : float
        The desired aspect ratio (width / height) of the cropped output.
        For example: 1.0 for square crop, 2.0 for wide crop, 0.5 for tall crop.
    relative_height : float
        A scaling factor for how much of the bounding box height to use as
        the crop height. 1.0 is 100% of face box height.

    Returns
    -------
    np.ndarray
        The cropped image as a NumPy array with shape approximately maintaining
        the requested aspect ratio. The crop will be clamped to image boundaries.

    Notes
    -----
        - If the computed crop goes beyond the image border, it is clamped to fit.
        - Normalized coordinates (0 to 1) are used for centroid and bounding box.
        - This method avoids jitter from changing box shapes and preserves
        face size consistency over time.
    """
    img_h, img_w, _ = frame.shape

    # Unpack normalized coordinates.
    cx_norm, cy_norm = centroid
    _, _, _, bbox_height_norm = bbox

    # Convert centroid to pixel coordinates.
    cx = int(cx_norm * img_w)
    cy = int(cy_norm * img_h)

    # Determine crop height in pixels based on bounding box height.
    crop_h = int(bbox_height_norm * img_h * relative_height)
    crop_w = int(crop_h * target_aspect_ratio)

    # Compute pixel crop box around centroid.
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(img_w, cx + crop_w // 2)
    y2 = min(img_h, cy + crop_h // 2)

    return frame[y1:y2, x1:x2]


def get_screen_resolution() -> tuple[int, int]:
    """
    Detects the current operating system (Raspbian, Ubuntu, MacOS)
    and returns the width and height of the primary monitor in pixels.

    Returns
    -------
    tuple[int, int]
        A tuple (width, height) representing screen resolution.
        Returns None if detection fails or unsupported platform.
    """
    system = platform.system()

    # MacOS
    if system == "Darwin":
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        )
        match = re.search(r"Resolution: (\d+) x (\d+)", output)
        if match:
            width, height = map(int, match.groups())

    elif system == "Linux":
        distro = platform.freedesktop_os_release().get("ID", "").lower()

        if "debian" in distro or "raspberrypi" in distro:
            # Raspbian: Use fbset
            output = subprocess.check_output(["fbset"], text=True)
            match = re.search(r"mode\s+\"(\d+)x(\d+)\"", output)
            if match:
                width, height = map(int, match.groups())

        elif "ubuntu" in distro:
            # Ubuntu: Use xrandr
            output = subprocess.check_output(["xrandr"], text=True)
            match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", output)
            if match:
                width, height = map(int, match.groups())

    return width, height


def generate_static(
    display_width: int,
    display_height: int,
    static_size : int
    ) -> np.ndarray:
    """
    Generates a frame of classic TV static with "pixels" that
    are static_size x static_size (in pixels).

    Parameters
    ----------
    display_width : int
        The width of the desired frame.
    display_height : int
        The height of the desired frame.
    static_size : int
        The height and width of each "static snowflake" in pixels.

    Returns
    -------
    np.ndarray
        A static frame.
    """
    h, w = display_height // static_size, display_width // static_size

    # Create small static noise image (grayscale).
    static_small = np.random.choice(
        [0, 255],
        size=(h, w),
        p=[0.5, 0.5]).astype('uint8')

    # Resize small image to full display size (still grayscale 2D).
    static_large = cv2.resize(
        static_small,
        (display_width, display_height),
        interpolation=cv2.INTER_NEAREST)

    # Convert to 3 channels by stacking grayscale image 3 times.
    static = np.stack([static_large]*3, axis=2)  # shape (H, W, 3)

    return static


def save_frames(
    last_cropped_frames : deque,
    folder: str,
    max_files : int = 10
    ) -> None:

    os.makedirs(folder, exist_ok=True)

    # Delete a random file if folder has too many.
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if len(files) >= max_files:
        file_to_delete = random.choice(files)
        os.remove(os.path.join(folder, file_to_delete))
        print(f"Deleted: {file_to_delete}")

    # Save new frames.
    filename = f"{uuid.uuid4()}.npy"
    filepath = os.path.join(folder, filename)
    # np.save(filepath, np.stack(last_cropped_frames))
    frames_to_save = list(last_cropped_frames)[:last_cropped_frames.maxlen]
    np.save(filepath, np.stack(frames_to_save))
    print(f"Saved: {filepath}")


def get_face_video(
    display_width: int,
    display_height: int,
    relative_height: float,
    smoothing: float,
    face_detection_confidence: float,
    recording_buffer_len: int,
    debug: bool
) -> None:
    """
    Starts webcam video capture and tracks a single face in real time.
    """
    # Buffer to store last 60 cropped frames
    recorded_frames: deque[np.ndarray] = deque(maxlen=recording_buffer_len)

    # Start video capture.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    # Initialize centroid and bounding box tracking.
    tracked_centroid: Optional[tuple[float, float]] = None
    prev_bbox: Optional[tuple[float, float, float, float]] = None

    # Initialize face detection.
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=face_detection_confidence)

    # Main event loop.
    while True:
        # Initialize the cropped frame.
        cropped = None

        # Capture a single frame.
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use mediapipe to process the frame.
        results = face_detection.process(rgb_frame)

        # If a face is detected, continue.
        if results.detections:  # type: ignore
            centroids = []
            bboxes = []

            for detection in results.detections:  # type: ignore
                bboxC = detection.location_data.relative_bounding_box
                bbox = (bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height)
                centroid = get_centroid(bbox)
                bboxes.append(bbox)
                centroids.append(centroid)

            if tracked_centroid is None:
                # First time: choose a face randomly
                idx = random.randint(0, len(centroids) - 1)
            else:
                # Track face closest to the previously tracked centroid
                distances = [euclidean_distance(c, tracked_centroid) 
                             for c in centroids]
                idx = int(np.argmin(distances))

            current_bbox = bboxes[idx]

            if tracked_centroid is None:
                tracked_centroid = centroids[idx]
            else:
                tracked_centroid = smooth_point(
                    tracked_centroid,
                    centroids[idx],
                    alpha=smoothing)

            if prev_bbox is None:
                smoothed_bbox = current_bbox
            elif significant_change(prev_bbox, current_bbox):
                smoothed_bbox = smooth_bbox(
                    prev_bbox,
                    current_bbox,
                    alpha=smoothing)
            else:
                # Hold previous box if no significant change
                smoothed_bbox = prev_bbox

            prev_bbox = smoothed_bbox

            if tracked_centroid is not None:
                cropped = crop_with_aspect_ratio(
                    frame=frame,
                    centroid=tracked_centroid,
                    bbox=smoothed_bbox,
                    target_aspect_ratio=display_width/display_height,
                    relative_height=relative_height)

                # Save cropped frame to buffer.
                display_image = cv2.resize(
                    cropped, 
                    (display_width, display_height),
                    interpolation=cv2.INTER_LINEAR
                )
                recorded_frames.appendleft(display_image)

        else:
            # Reset tracking if no face is detected.
            tracked_centroid = None
            prev_bbox = None

            # If there is a break in the detection streak (else condition)
            # and enough frames have accumulated, write to disk!
            if (len(recorded_frames) >= recording_buffer_len):
                save_frames(
                    recorded_frames,
                    folder="videos",
                    max_files=10)

                # Reset the recorded frames.
                recorded_frames.clear()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()


def display_random_videos():
    cv2.namedWindow("Random Video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Random Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    folder = "videos"

    # Load first video
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        print("No videos found.")
        return
    current_file = os.path.join(folder, random.choice(files))
    current_frames = np.load(current_file)

    next_frames = None
    preload_thread = None

    # Start preloading next video immediately
    def start_preload():
        nonlocal next_frames
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if not files:
            next_frames = None
            return
        next_file = os.path.join(folder, random.choice(files))
        try:
            next_frames = np.load(next_file)
        except Exception as e:
            print("Failed to preload:", e)
            next_frames = None

    preload_thread = threading.Thread(target=start_preload)
    preload_thread.start()

    while True:
        for frame in current_frames:
            cv2.imshow("Random Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Swap to preloaded frames immediately
        if preload_thread is not None:
            preload_thread.join()  # only wait here if preload is still running
        if next_frames is not None:
            current_frames = next_frames
        else:
            # If preload failed, reload a random video
            files = [f for f in os.listdir(folder) if f.endswith(".npy")]
            if not files:
                continue
            current_frames = np.load(os.path.join(folder, random.choice(files)))

        # Immediately start preloading the next video
        next_frames = None
        preload_thread = threading.Thread(target=start_preload)
        preload_thread.start()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Detect which system is being used and get the screen resolution.
    display_width, display_height = get_screen_resolution()

    # Start the video recording function as a background thread.
    video_collector = threading.Thread(
        target=get_face_video,
        args=(
            display_width,
            display_height,
            config["relative_height"],
            config["smoothing"],
            config["face_detection_confidence"],
            config["recording_buffer_len"],
            config["debug"]))
    
    video_collector.start()

    # Start the video player.
    display_random_videos()
