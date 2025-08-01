import cv2
import face_recognition
import numpy as np
import os



def capture_image_from_webcam() -> np.ndarray:
    """
    Captures a single frame from the webcam.

    Parameters
    ----------
    None
        Snaps a picture.
    
    Returns
    -------
    np.ndarray
        A picture from the webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Press q to capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb


def get_face_embedding(image : np.ndarray) -> np.ndarray | None:
    """
    Detects a face in the input image and returns its 128-dimension embedding.

    This assumes that there is exactly one face in the image; if multiple are
    detected, only the first one is used.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array in RGB format.

    Returns
    -------
    np.ndarray
        A 1D array of shape (128,) representing the face embedding.
    None
        If no face is detected or encoding fails.
    """
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print("No face detected.")
        return None

    print(f"{len(face_locations)} face(s) detected.")
    embeddings = face_recognition.face_encodings(image, face_locations)
    
    return embeddings[0]


def save_embedding(embedding : np.ndarray,
                   filename : str) -> None:
    """
    Saves a face embedding to disk as a .npy file.

    Parameters
    ----------
    embedding : np.ndarray
        A 1D array of shape (128,) representing the face embedding.
    filename : str
        The path (including filename) where the .npy file will be saved.
        Example: "embeddings/face_01.npy"
    """
    # Make directory path if it doesn't already exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the embedding.
    np.save(filename, embedding)


def load_embedding(filename : str) -> np.ndarray:
    """
    Loads a .npy file containing the embedding for a single face.

    Parameters
    ----------
    filename : str
        The path to a .npy file containing a single face embedding.

    Returns
    -------
    np.ndarray
        A 1D array of shape (128,) representing the face embedding.
    """
    embedding = np.load(filename)

    return embedding



if __name__ == "__main__":
    """
    Get a pictue of the Buddha statue so that it can be ignored.

    First point the webcam at Buddha. Make sure no other faces are
    in the image and lighting will match the lighting conditions
    during runtime.
    """
    # Get a picture of Buddha.
    picture = capture_image_from_webcam()

    # Get the face embedding.
    embedding = get_face_embedding(picture)

    # Save it. This is all we need!
    save_embedding(embedding, filename="buddha_embedding.npy")

    # Test out the loading function.
    embedding_loaded = load_embedding(filename="buddha_embedding.npy")
