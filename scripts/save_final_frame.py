import cv2
import os

def save_frame_10():
    video_path = 'demo_material/demo_cam34.avi'
    output_path = 'docs/system_output.jpg'
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    # Frame 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Successfully saved frame 10 to {output_path}")
    else:
        print("Error: Could not read frame 10.")
    
    cap.release()

if __name__ == "__main__":
    save_frame_10()
