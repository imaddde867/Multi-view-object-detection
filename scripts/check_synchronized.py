import cv2
import numpy as np

def motion_curve(video):
    cap = cv2.VideoCapture(video)
    prev = None
    curve = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = gray
            continue

        diff = cv2.absdiff(prev, gray)
        motion = np.sum(diff)
        curve.append(motion)

        prev = gray

    cap.release()
    return np.array(curve)

curve1 = motion_curve("data/raw/testing_videos/Cam1.mp4")
curve2 = motion_curve("data/raw/testing_videos/Cam2.mp4")

# Find best alignment using cross-correlation
offset = np.argmax(np.correlate(curve1, curve2, mode="full")) - len(curve2)
print("Estimated time offset (in frames):", offset)
